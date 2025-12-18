"""
Kappa Estimation Experiment: Mean Field d-Scaling

Estimates the spatial correction factor κ from:
    n = (b-d)/d' - κ*d/d'

When d=0: n = b/d' (mean field, no spatial structure)
When kernels are equal (birth_std = death_std), we expect small κ.

This script:
1. Runs simulations for various d values in 1D periodic domain
2. Computes first moment (density) and second moment g(r) via FFT
3. Handles autocorrelation via batch means
4. Saves results to numpy and generates plots
"""
from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from joblib import Parallel, delayed
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from SSA.numba_sim_normal import make_normal_ssa_1d, SSANormalState
from SSA.experiments.configs import ExperimentConfig, get_config, list_configs, baseline_config


# =============================================================================
# PCF Calculation via FFT
# =============================================================================
def compute_pcf_fft_1d(positions: NDArray[np.float64], L: float, 
                       dr: float, r_max: float) -> tuple[NDArray, NDArray]:
    """
    FFT-based pair correlation function for 1D periodic domain.
    
    Returns (r_bins, g_r) where g(r) is normalized pair correlation.
    
    The normalization uses N*(N-1) for distinct pairs (not N²),
    which is important for getting g(r)=1 for Poisson process.
    """
    N = len(positions)
    if N < 2:
        n_r = int(r_max / dr) + 1
        return np.arange(n_r) * dr, np.ones(n_r)
    
    n_bins = int(L / dr)
    
    # Bin particles into density field using numpy histogram
    density, _ = np.histogram(positions, bins=n_bins, range=(0, L))
    density = density.astype(np.float64)
    
    # FFT for pair correlation via convolution theorem
    # This computes C(r) = sum_i sum_j indicator(x_i - x_j in bin r)
    # which includes self-pairs (i=j) at r=0
    rho_hat = np.fft.fft(density)
    power = np.abs(rho_hat)**2
    corr = np.fft.ifft(power).real
    
    # Normalize by DISTINCT pairs N*(N-1), not N²
    # For uniform/Poisson: expected pairs in bin dr at distance r is:
    #   N*(N-1) * (dr/L) = N*(N-1)/L * dr
    # So g(r) = C(r) / (N*(N-1)/L * dr) = C(r) * L / (N*(N-1) * dr)
    g_r = corr * L / (N * (N - 1) * dr)
    
    # Extract bins up to r_max
    n_r_bins = min(int(r_max / dr) + 1, n_bins // 2)
    r = np.arange(n_r_bins) * dr
    return r, g_r[:n_r_bins]


# =============================================================================
# Batch Statistics for MCMC-style handling
# =============================================================================
def compute_batch_statistics(samples: NDArray[np.float64], n_batches: int
                            ) -> tuple[float, float, NDArray]:
    """
    Compute statistics using batch means for autocorrelated samples.
    
    Returns: (mean, std_error, batch_means)
    """
    samples = np.asarray(samples)
    batch_size = len(samples) // n_batches
    if batch_size < 1:
        return samples.mean(), samples.std() / np.sqrt(len(samples)), samples
    
    batch_means = np.array([
        samples[i*batch_size:(i+1)*batch_size].mean() 
        for i in range(n_batches)
    ])
    
    mean = batch_means.mean()
    std_err = batch_means.std() / np.sqrt(n_batches)
    return mean, std_err, batch_means


def estimate_autocorrelation_time(samples: NDArray[np.float64], 
                                  max_lag: int = 100) -> float:
    """Estimate integrated autocorrelation time."""
    samples = np.asarray(samples)
    n = len(samples)
    if n < max_lag:
        max_lag = n // 2
    
    mean = samples.mean()
    var = samples.var()
    if var < 1e-12:
        return 1.0
    
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        c = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean))
        autocorr[lag] = c / var
    
    tau = 1.0
    for k in range(1, max_lag):
        if autocorr[k] < 0.05:
            break
        tau += 2 * autocorr[k]
    
    return tau


# =============================================================================
# Single Simulation Run
# =============================================================================
def run_single_d(d_val: float, cfg: ExperimentConfig, seed: int,
                 progress_callback: Callable | None = None,
                 collect_diagnostics: bool = False) -> dict:
    """
    Run full simulation for a single d value.
    
    Returns dict with density statistics and g(r) with batch-level data.
    """
    n_exp = cfg.n_expected(d_val)
    pop_exp = cfg.pop_expected(d_val)
    
    if pop_exp < 10:
        n_r = int(cfg.r_max / cfg.dr) + 1
        return {
            'd': d_val, 'n_expected': n_exp, 'pop_expected': pop_exp,
            'density_mean': np.nan, 'density_std': np.nan,
            'g_r': None, 'g_r_batches': None, 'r_bins': np.arange(n_r) * cfg.dr,
            'warmup_time': 0.0, 'measurement_time': 0.0, 'total_events': 0,
            'pop_trace': None, 'phase_markers': None, 'warmup_end_positions': None
        }
    
    # Create simulator
    sim = make_normal_ssa_1d(
        M=1,
        area_len=cfg.L,
        birth_rates=[cfg.b],
        death_rates=[d_val],
        dd_matrix=[[cfg.d_prime]],
        birth_std=[cfg.sigma],
        death_std=[[cfg.sigma]],
        death_cull_sigmas=5.0,
        is_periodic=True,
        seed=seed,
    )
    
    # === Warmup with trace tracking ===
    warmup_start = time.perf_counter()
    
    # Only collect traces if diagnostics enabled
    pop_trace_t = [] if collect_diagnostics else None
    pop_trace_n = [] if collect_diagnostics else None
    
    # Phase 1: Start at 10%, burst to 20%
    initial_pop = max(10, int(cfg.initial_density_frac * pop_exp))
    sim.spawn_random(0, initial_pop)
    
    target_20 = int(0.2 * pop_exp)
    t_sim_start = sim.current_time()
    if collect_diagnostics:
        pop_trace_t.append(sim.current_time())
        pop_trace_n.append(sim.current_population())
    
    while sim.current_population() < target_20:
        sim.run_events(1000)
        if sim.current_population() <= 0:
            sim.spawn_random(0, initial_pop)
        if collect_diagnostics:
            pop_trace_t.append(sim.current_time())
            pop_trace_n.append(sim.current_population())
    
    t_20 = sim.current_time() - t_sim_start
    t_phase1_end = sim.current_time()
    if t_20 < 1e-9:
        t_20 = 1.0  # Fallback
    
    # Phase 2: Time steps until equilibrium (no significant difference between windows)
    pop_history = [sim.current_population()]
    while True:
        sim.run_until_time(t_20)
        curr_pop = sim.current_population()
        if collect_diagnostics:
            pop_trace_t.append(sim.current_time())
            pop_trace_n.append(curr_pop)
        pop_history.append(curr_pop)
        
        # Check for equilibrium: compare last 10 vs previous 10
        if len(pop_history) >= 20:
            recent = np.array(pop_history[-10:])     # Last 10 measurements
            earlier = np.array(pop_history[-20:-10]) # Previous 10 measurements
            
            recent_mean = recent.mean()
            earlier_mean = earlier.mean()
            pooled_std = np.sqrt((recent.var() + earlier.var()) / 2)
            
            # No significant difference: means within 1 std of each other
            if pooled_std > 0:
                diff_ratio = abs(recent_mean - earlier_mean) / pooled_std
                if diff_ratio < 1.0:
                    break
            else:
                # Zero variance means equilibrium
                break
    t_phase2_end = sim.current_time()
    
    # Phase 3: 100% extra warmup time (double total time once negative growth)
    total_warmup_sim_time = sim.current_time() - t_sim_start
    sim.run_until_time(total_warmup_sim_time)  # Run for same amount of time again
    if collect_diagnostics:
        pop_trace_t.append(sim.current_time())
        pop_trace_n.append(sim.current_population())
    t_warmup_end = sim.current_time()
    
    # Record positions at warmup end for density histogram (only if diagnostics enabled)
    warmup_end_positions = None
    if collect_diagnostics:
        warmup_pop = sim.current_population()
        warmup_end_positions = np.array([sim.positions[i, 0] for i in range(warmup_pop)])
    
    warmup_time = time.perf_counter() - warmup_start
    
    # === Measurement with Online Batch Statistics ===
    # Memory optimization: compute batch statistics incrementally instead of 
    # storing all samples (reduces memory from O(total_samples * n_r_bins) to O(n_batches * n_r_bins))
    measure_start = time.perf_counter()
    
    batch_size = cfg.samples_per_batch
    n_batches = cfg.n_batches
    n_events_per_sample = max(10, int(cfg.event_frac * pop_exp))
    total_events = 0
    
    # Initialize accumulators - only store batch-level statistics
    r_bins = None
    n_r_bins = None
    g_r_batches = None  # [n_batches, n_r_bins] - batch means
    density_batch_sums = np.zeros(n_batches, dtype=np.float64)
    
    # Only store all density samples if diagnostics enabled (needed for autocorrelation analysis)
    density_samples_list = [] if collect_diagnostics else None
    
    # Current batch accumulator
    current_batch_g_r_sum = None
    current_batch_sample_count = 0
    current_batch_idx = 0
    
    for sample_idx in range(cfg.total_samples):
        sim.run_events(n_events_per_sample)
        total_events += n_events_per_sample
        
        pop = sim.current_population()
        density = pop / cfg.L
        
        # Accumulate density for current batch
        density_batch_sums[current_batch_idx] += density
        if collect_diagnostics:
            density_samples_list.append(density)
        
        # Compute PCF
        positions = np.array([sim.positions[i, 0] for i in range(pop)])
        r, gr = compute_pcf_fft_1d(positions, cfg.L, cfg.dr, cfg.r_max)
        
        # Initialize batch storage on first sample
        if r_bins is None:
            r_bins = r
            n_r_bins = len(r)
            g_r_batches = np.zeros((n_batches, n_r_bins), dtype=np.float64)
            current_batch_g_r_sum = np.zeros(n_r_bins, dtype=np.float64)
        
        # Accumulate g(r) for current batch
        current_batch_g_r_sum += gr
        current_batch_sample_count += 1
        
        # Check if batch is complete
        if current_batch_sample_count >= batch_size:
            # Store batch mean
            g_r_batches[current_batch_idx] = current_batch_g_r_sum / batch_size
            
            # Reset for next batch
            current_batch_idx += 1
            current_batch_sample_count = 0
            current_batch_g_r_sum = np.zeros(n_r_bins, dtype=np.float64)
    
    measurement_time = time.perf_counter() - measure_start
    
    # Compute final statistics from batch means
    # Average g(r) across all batches
    g_r_mean = g_r_batches.mean(axis=0)
    
    # Density statistics from batch sums
    density_batch_means = density_batch_sums / batch_size
    density_mean = density_batch_means.mean()
    density_std = density_batch_means.std() / np.sqrt(n_batches)
    
    # Build result dict
    result = {
        'd': d_val,
        'n_expected': n_exp,
        'pop_expected': pop_exp,
        'density_mean': density_mean,
        'density_std': density_std,
        'batch_means': density_batch_means,
        'g_r': g_r_mean,
        'g_r_batches': g_r_batches,
        'r_bins': r_bins,
        'warmup_time': warmup_time,
        'measurement_time': measurement_time,
        'total_events': total_events,
        # Phase markers always included (small data)
        'phase_markers': {
            't_start': t_sim_start,
            't_phase1_end': t_phase1_end,
            't_phase2_end': t_phase2_end,
            't_warmup_end': t_warmup_end,
        },
    }
    
    # Add diagnostic data only if requested (large arrays)
    if collect_diagnostics:
        result['density_samples'] = np.array(density_samples_list)
        result['pop_trace_t'] = np.array(pop_trace_t)
        result['pop_trace_n'] = np.array(pop_trace_n)
        result['warmup_end_positions'] = warmup_end_positions
    else:
        result['density_samples'] = None
        result['pop_trace_t'] = None
        result['pop_trace_n'] = None
        result['warmup_end_positions'] = None
    
    return result


# =============================================================================
# Main Experiment Runner
# =============================================================================
def run_experiment(cfg: ExperimentConfig, test_mode: bool = False, 
                   collect_diagnostics: bool = False) -> dict:
    """
    Run full experiment across all d values.
    
    Args:
        cfg: Experiment configuration
        test_mode: If True, only run d=0 case
        collect_diagnostics: If True, collect detailed diagnostic data (increases memory usage)
    """
    if test_mode:
        d_values = np.array([0.0])
        print("=== TEST MODE: Running mean-field case only ===")
    else:
        d_values = cfg.get_d_values()
    
    print(f"Config: {cfg.name}")
    print(f"Domain: L={cfg.L}, σ={cfg.sigma}")
    print(f"Expected pop for d=0: {cfg.pop_expected(0)}")
    print(f"Samples per d: {cfg.total_samples} ({cfg.n_batches} batches × {cfg.samples_per_batch})")
    print(f"Events per sample: {int(cfg.event_frac * cfg.pop_expected(0))}")
    print(f"PCF: dr={cfg.dr}, r_max={cfg.r_max}")
    print(f"d values: {len(d_values)} cases")
    print(f"Diagnostics: {'enabled' if collect_diagnostics else 'disabled (use --diagnostics to enable)'}")
    print()
    
    results_list = Parallel(n_jobs=cfg.n_jobs if not test_mode else 1)(
        delayed(run_single_d)(d, cfg, seed=42 + i, collect_diagnostics=collect_diagnostics)
        for i, d in enumerate(tqdm(d_values, desc="Running simulations"))
    )
    
    results = {
        'config': cfg,
        'd_values': d_values,
        'density_means': np.array([r['density_mean'] for r in results_list]),
        'density_stds': np.array([r['density_std'] for r in results_list]),
        'n_expected': np.array([r['n_expected'] for r in results_list]),
        'warmup_times': np.array([r['warmup_time'] for r in results_list]),
        'measurement_times': np.array([r['measurement_time'] for r in results_list]),
        'total_events': np.array([r['total_events'] for r in results_list]),
        'g_r_all': [r['g_r'] for r in results_list],
        'g_r_batches_all': [r['g_r_batches'] for r in results_list],
        'r_bins': results_list[0]['r_bins'] if results_list[0]['r_bins'] is not None else None,
        'full_results': results_list,
    }
    
    return results


# =============================================================================
# Saving and Plotting
# =============================================================================
def save_results(results: dict, cfg: ExperimentConfig, filename: str = "kappa_results.npz"):
    """Save results to numpy file."""
    output_dir = Path(cfg.get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    g_r_stack = np.array([gr if gr is not None else np.full(len(results['r_bins']), np.nan) 
                         for gr in results['g_r_all']])
    
    # Also save batch data for PCF
    g_r_batches_stack = np.array([
        grb if grb is not None else np.full((cfg.n_batches, len(results['r_bins'])), np.nan)
        for grb in results['g_r_batches_all']
    ])  # [n_d, n_batches, n_r_bins]
    
    np.savez(
        output_dir / filename,
        d_values=results['d_values'],
        density_means=results['density_means'],
        density_stds=results['density_stds'],
        n_expected=results['n_expected'],
        warmup_times=results['warmup_times'],
        measurement_times=results['measurement_times'],
        r_bins=results['r_bins'],
        g_r_all=g_r_stack,
        g_r_batches_all=g_r_batches_stack,
        L=cfg.L,
        b=cfg.b,
        d_prime=cfg.d_prime,
        sigma=cfg.sigma,
        dr=cfg.dr,
        n_batches=cfg.n_batches,
    )
    print(f"Results saved to {output_dir / filename}")


def plot_results(results: dict, cfg: ExperimentConfig, save_plots: bool = True, show_plots: bool = True):
    """Generate and optionally save plots."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(cfg.get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    d_vals = results['d_values']
    n_meas = results['density_means']
    n_err = results['density_stds']
    n_exp = results['n_expected']
    
    z_95 = 1.96  # 95% confidence interval
    
    # --- Plot 1: Density vs d ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.errorbar(d_vals, n_meas, yerr=z_95*n_err, fmt='o-', capsize=3, label='Measured (95% CI)')
    ax.plot(d_vals, n_exp, 'r--', label='Mean field: (b-d)/d\'', linewidth=2)
    ax.set_xlabel('d (intrinsic death rate)')
    ax.set_ylabel('Equilibrium density n')
    ax.set_title(f'First Moment: Density vs d (σ={cfg.sigma})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    diff = n_meas - n_exp
    ax.errorbar(d_vals, diff, yerr=z_95*n_err, fmt='o-', capsize=3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('d (intrinsic death rate)')
    ax.set_ylabel('n_measured - n_expected')
    ax.set_title('Deviation from Mean Field (95% CI)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_dir / 'density_vs_d.png', dpi=150)
        print(f"Saved: {output_dir / 'density_vs_d.png'}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    # --- Plot 2: g(r) for different d values ---
    if results['r_bins'] is not None:
        r = results['r_bins']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        r_plot = r[1:]  # Exclude r=0
        
        cmap = plt.cm.viridis
        d_min, d_max = d_vals.min(), d_vals.max()
        norm = plt.Normalize(vmin=d_min, vmax=d_max) if d_max > d_min else plt.Normalize(0, 1)
        
        for i, d in enumerate(d_vals):
            gr = results['g_r_all'][i]
            if gr is not None:
                ax.plot(r_plot, gr[1:], color=cmap(norm(d)), alpha=0.8, linewidth=1.5)
        
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='Poisson')
        ax.set_xlabel('r')
        ax.set_ylabel('g(r)')
        ax.set_title(f'Pair Correlation Function (σ={cfg.sigma})')
        ax.set_xlim(r_plot[0], cfg.r_max)
        ax.grid(True, alpha=0.3)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('d (intrinsic death rate)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / 'pcf_g_r.png', dpi=150)
            print(f"Saved: {output_dir / 'pcf_g_r.png'}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    # --- Plot 3: PCF values at specific r vs d with confidence bands ---
    plot_pcf_vs_d(results, cfg, save_plots, show_plots)
    
    # --- Plot 4: Kappa extraction (linear fit) ---
    if len(d_vals) > 3:
        # Use fit_d_max from config to limit fitting range
        fit_d_max = cfg.fit_d_max
        mask = (d_vals > 0.0005) & (d_vals <= fit_d_max)  # Exclude d=0 and very small d, limit to fit_d_max
        
        if mask.sum() > 2:
            x = d_vals[mask]
            y = (n_exp[mask] - n_meas[mask]) * cfg.d_prime
            y_err = n_err[mask] * cfg.d_prime  # Propagate errors
            
            # Weighted linear fit through origin: y = kappa * x
            # Weighted least squares: kappa = sum(w*x*y) / sum(w*x^2) where w = 1/var
            weights = 1.0 / (y_err**2 + 1e-12)  # Add small epsilon to avoid div by zero
            kappa_est = np.sum(weights * x * y) / np.sum(weights * x**2)
            
            # Standard error of kappa estimate
            kappa_var = 1.0 / np.sum(weights * x**2)
            kappa_err = np.sqrt(kappa_var)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot all points (gray for points outside fit range)
            all_mask = d_vals > 0.0005
            outside_mask = all_mask & (d_vals > fit_d_max)
            x_all = d_vals[all_mask]
            y_all = (n_exp[all_mask] - n_meas[all_mask]) * cfg.d_prime
            y_err_all = n_err[all_mask] * cfg.d_prime
            
            # Points outside fit range (gray)
            if outside_mask.sum() > 0:
                x_out = d_vals[outside_mask]
                y_out = (n_exp[outside_mask] - n_meas[outside_mask]) * cfg.d_prime
                y_err_out = n_err[outside_mask] * cfg.d_prime
                ax.errorbar(x_out, y_out, yerr=1.96*y_err_out, fmt='o', color='gray', 
                           alpha=0.5, capsize=3, markersize=6, label='Outside fit range')
            
            # Points inside fit range (blue with error bars)
            ax.errorbar(x, y, yerr=1.96*y_err, fmt='o', color='C0', 
                       capsize=3, markersize=8, label=f'Used for fit (d ≤ {fit_d_max})')
            
            # Fit line with confidence band
            x_fit = np.linspace(0, d_vals.max() * 1.1, 100)
            y_fit = kappa_est * x_fit
            y_fit_upper = (kappa_est + 1.96*kappa_err) * x_fit
            y_fit_lower = (kappa_est - 1.96*kappa_err) * x_fit
            
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'κ = {kappa_est:.4f} ± {1.96*kappa_err:.4f}')
            ax.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.15)
            
            # Mark the fit range
            ax.axvline(fit_d_max, color='green', linestyle='--', alpha=0.5, 
                      label=f'fit_d_max = {fit_d_max}')
            
            ax.set_xlabel('d')
            ax.set_ylabel('(n_expected - n_measured) × d\'')
            ax.set_title(f'Kappa Extraction: κ = {kappa_est:.4f} ± {1.96*kappa_err:.4f} (σ={cfg.sigma})')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(output_dir / 'kappa_extraction.png', dpi=150)
                print(f"Saved: {output_dir / 'kappa_extraction.png'}")
            if show_plots:
                plt.show()
            else:
                plt.close(fig)


def plot_pcf_vs_d(results: dict, cfg: ExperimentConfig, save_plots: bool = True, show_plots: bool = True):
    """
    Plot PCF values at first 6 r values in a 2x3 grid vs d 
    with 95% confidence bands using batch statistics.
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(cfg.get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results['r_bins'] is None:
        return
    
    r_bins = results['r_bins']
    d_vals = results['d_values']
    g_r_batches_all = results['g_r_batches_all']
    
    # Use first 6 non-zero r bins (indices 1-6)
    n_plots = 6
    start_idx = 1  # Skip r=0
    
    if len(r_bins) < start_idx + n_plots:
        n_plots = max(0, len(r_bins) - start_idx)
    
    if n_plots == 0:
        print("Warning: Not enough r bins for PCF grid plot")
        return
    
    # Create 2x3 grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    z_95 = 1.96  # 95% confidence interval
    
    for plot_idx in range(n_plots):
        ax = axes[plot_idx]
        bin_idx = start_idx + plot_idx
        actual_r = r_bins[bin_idx]
        
        g_means = []
        g_stds = []
        
        for i, d in enumerate(d_vals):
            batches = g_r_batches_all[i]
            if batches is None:
                g_means.append(np.nan)
                g_stds.append(np.nan)
                continue
            
            # Get g(r) values at this r_bin for all batches
            batch_values = batches[:, bin_idx]
            
            # Compute mean and std error
            mean = batch_values.mean()
            std_err = batch_values.std() / np.sqrt(len(batch_values))
            
            g_means.append(mean)
            g_stds.append(std_err)
        
        g_means = np.array(g_means)
        g_stds = np.array(g_stds)
        
        # Plot with error bars (more visible) and confidence band
        ax.errorbar(d_vals, g_means, yerr=z_95*g_stds, fmt='o-', color='C0', 
                   markersize=6, capsize=4, capthick=1.5, elinewidth=1.5, label='95% CI')
        ax.fill_between(d_vals, g_means - z_95*g_stds, g_means + z_95*g_stds, 
                       color='C0', alpha=0.15)
        
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Poisson')
        ax.set_xlabel('d', fontsize=10)
        ax.set_ylabel('g(r)', fontsize=10)
        ax.set_title(f'r = {actual_r:.2f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'PCF g(r) vs d at Different r Values (σ={cfg.sigma})', fontsize=12)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_dir / 'pcf_vs_d.png', dpi=150)
        print(f"Saved: {output_dir / 'pcf_vs_d.png'}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Diagnostic Plots
# =============================================================================
def plot_diagnostics(results: dict, cfg: ExperimentConfig, show_plots: bool = False):
    """
    Generate diagnostic plots for warmup and initial state.
    Saves to experiments/diagnostics/{config_name}/ folder.
    """
    import matplotlib.pyplot as plt
    
    diag_dir = Path(__file__).parent / "diagnostics" / cfg.name
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    full_results = results['full_results']
    d_vals = results['d_values']
    n_d = len(d_vals)
    
    # Plot in groups of 6 (2x3 grid)
    d_per_plot = 6
    n_plots = (n_d + d_per_plot - 1) // d_per_plot
    
    # --- Population trace plots (6 d values per figure) ---
    for plot_idx in range(n_plots):
        start_i = plot_idx * d_per_plot
        end_i = min(start_i + d_per_plot, n_d)
        n_subplots = end_i - start_i
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for subplot_idx, i in enumerate(range(start_i, end_i)):
            ax = axes[subplot_idx]
            r = full_results[i]
            d_val = d_vals[i]
            
            if r.get('pop_trace_t') is None:
                ax.text(0.5, 0.5, f'd={d_val:.4f}\n(no data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'd = {d_val:.4f}')
                continue
            
            t = r['pop_trace_t']
            n = r['pop_trace_n']
            markers = r['phase_markers']
            pop_exp = r['pop_expected']
            
            # Plot population trace
            ax.plot(t, n, 'b-', linewidth=1, alpha=0.8, label='Population')
            
            # Add phase markers
            t_start = markers['t_start']
            t_p1 = markers['t_phase1_end']
            t_p2 = markers['t_phase2_end']
            t_end = markers['t_warmup_end']
            
            ax.axvline(t_p1, color='orange', linestyle='--', alpha=0.7, label='Phase 1 end (20%)')
            ax.axvline(t_p2, color='red', linestyle='--', alpha=0.7, label='Phase 2 end (neg growth)')
            ax.axvline(t_end, color='green', linestyle='--', alpha=0.7, label='Warmup end')
            
            # Add expected population line
            ax.axhline(pop_exp, color='gray', linestyle=':', alpha=0.5, label=f'Expected: {pop_exp}')
            ax.axhline(0.2 * pop_exp, color='orange', linestyle=':', alpha=0.3)
            
            ax.set_xlabel('Simulation time')
            ax.set_ylabel('Population')
            ax.set_title(f'd = {d_val:.4f}')
            ax.grid(True, alpha=0.3)
            
            if subplot_idx == 0:
                ax.legend(loc='best', fontsize=7)
        
        # Hide unused axes
        for idx in range(n_subplots, 6):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Population Warmup Traces ({cfg.name})', fontsize=14)
        plt.tight_layout()
        
        filename = f'pop_trace_grid_{plot_idx+1}.png'
        plt.savefig(diag_dir / filename, dpi=150)
        print(f"Saved: {diag_dir / filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    # --- Density histogram plots (6 d values per figure) ---
    for plot_idx in range(n_plots):
        start_i = plot_idx * d_per_plot
        end_i = min(start_i + d_per_plot, n_d)
        n_subplots = end_i - start_i
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for subplot_idx, i in enumerate(range(start_i, end_i)):
            ax = axes[subplot_idx]
            r = full_results[i]
            d_val = d_vals[i]
            
            positions = r.get('warmup_end_positions')
            if positions is None or len(positions) == 0:
                ax.text(0.5, 0.5, f'd={d_val:.4f}\n(no data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'd = {d_val:.4f}')
                continue
            
            # Plot histogram of positions (density estimate)
            n_bins = min(100, int(cfg.L / cfg.sigma))
            ax.hist(positions, bins=n_bins, range=(0, cfg.L), 
                   density=True, alpha=0.7, color='steelblue', edgecolor='none')
            
            # Add expected uniform density
            expected_density = 1.0 / cfg.L
            ax.axhline(expected_density, color='red', linestyle='--', 
                      alpha=0.7, label=f'Uniform: {expected_density:.4f}')
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Density')
            ax.set_title(f'd = {d_val:.4f}, N = {len(positions)}')
            ax.set_xlim(0, cfg.L)
            ax.grid(True, alpha=0.3)
            
            if subplot_idx == 0:
                ax.legend(loc='best', fontsize=8)
        
        # Hide unused axes
        for idx in range(n_subplots, 6):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Spatial Density at Warmup End ({cfg.name})', fontsize=14)
        plt.tight_layout()
        
        filename = f'density_hist_grid_{plot_idx+1}.png'
        plt.savefig(diag_dir / filename, dpi=150)
        print(f"Saved: {diag_dir / filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    print(f"Diagnostics saved to: {diag_dir}")


def print_test_summary(results: dict, cfg: ExperimentConfig):
    """Print summary for test run."""
    print("\n" + "="*60)
    print(f"TEST RUN SUMMARY (Mean Field Case: d=0, σ={cfg.sigma})")
    print("="*60)
    
    r = results['full_results'][0]
    
    z_95 = 1.96
    ci_95 = z_95 * r['density_std']
    print(f"\nExpected density: {r['n_expected']:.4f}")
    print(f"Measured density: {r['density_mean']:.4f} ± {ci_95:.4f} (95% CI)")
    print(f"Relative error: {abs(r['density_mean'] - r['n_expected']) / r['n_expected'] * 100:.2f}%")
    
    print(f"\nWarmup time: {r['warmup_time']:.2f} s")
    print(f"Measurement time: {r['measurement_time']:.2f} s")
    print(f"Total time: {r['warmup_time'] + r['measurement_time']:.2f} s")
    
    print(f"\nTotal events: {r['total_events']:,}")
    if r['measurement_time'] > 0:
        print(f"Events/second: {r['total_events'] / r['measurement_time']:.0f}")
    
    # Autocorrelation analysis only available with diagnostics enabled
    samples = r.get('density_samples')
    if samples is not None:
        tau = estimate_autocorrelation_time(samples)
        ess = len(samples) / tau
        print(f"\nAutocorrelation time τ ≈ {tau:.1f}")
        print(f"Effective sample size: {ess:.0f} / {len(samples)}")
    else:
        print("\n(Autocorrelation analysis requires --diagnostics flag)")
    
    n_d_full = len(cfg.get_d_values())
    time_per_d = r['warmup_time'] + r['measurement_time']
    print(f"\n--- Extrapolation for full run ({n_d_full} d-values) ---")
    print(f"Estimated total time (serial): {time_per_d * n_d_full / 60:.1f} min")
    
    n_cores = os.cpu_count() or 4
    print(f"Estimated total time ({n_cores} cores): {time_per_d * n_d_full / n_cores / 60:.1f} min")


# =============================================================================
# Entry Point
# =============================================================================
def run_single_config(config_name: str, test_mode: bool = False, 
                      collect_diagnostics: bool = False):
    """Run experiment for a single config."""
    cfg = get_config(config_name)
    print(f"\n{'='*60}")
    print(f"Configuration: {cfg.name}")
    print(f"  sigma={cfg.sigma}, d'={cfg.d_prime}, L={cfg.L}")
    print(f"  dr={cfg.dr}, r_max={cfg.r_max}")
    print('='*60)
    
    if test_mode:
        print("\nRunning test with mean-field case (d=0)...")
        results = run_experiment(cfg, test_mode=True, collect_diagnostics=collect_diagnostics)
        print_test_summary(results, cfg)
        save_results(results, cfg, filename="test_results.npz")
        plot_results(results, cfg, show_plots=True)
        if collect_diagnostics:
            plot_diagnostics(results, cfg, show_plots=False)
    else:
        print("\nRunning full experiment...")
        results = run_experiment(cfg, test_mode=False, collect_diagnostics=collect_diagnostics)
        save_results(results, cfg)
        plot_results(results, cfg, show_plots=False)  # Don't block for full runs
        if collect_diagnostics:
            plot_diagnostics(results, cfg, show_plots=False)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Kappa Estimation Experiment")
    parser.add_argument('--config', '-c', type=str, default='baseline',
                       choices=list_configs(),
                       help=f"Configuration to use. Available: {list_configs()}")
    parser.add_argument('--all', '-a', action='store_true',
                       help="Run all available configurations")
    parser.add_argument('--test', '-t', action='store_true',
                       help="Run test mode only (d=0)")
    parser.add_argument('--no-prompt', '-y', action='store_true',
                       help="Skip confirmation prompt, run full experiment")
    parser.add_argument('--diagnostics', '-d', action='store_true',
                       help="Collect diagnostic data (population traces, density histograms). "
                            "Warning: significantly increases memory usage!")
    
    args = parser.parse_args()
    
    # Run all configs mode
    if args.all:
        configs = list_configs()
        print(f"Running {'test' if args.test else 'full'} experiment for all {len(configs)} configs:")
        for name in configs:
            print(f"  - {name}")
        if args.diagnostics:
            print("  (with diagnostics enabled)")
        
        if not args.no_prompt and not args.test:
            response = input("\nProceed with all configs? [y/N]: ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
        
        for config_name in configs:
            run_single_config(config_name, test_mode=args.test, 
                            collect_diagnostics=args.diagnostics)
        
        print(f"\n{'='*60}")
        print(f"All {len(configs)} configurations completed!")
        print('='*60)
        return
    
    # Single config mode
    cfg = get_config(args.config)
    print(f"Using configuration: {cfg.name}")
    print(f"  sigma={cfg.sigma}, dr={cfg.dr}, r_max={cfg.r_max}")
    print()
    
    # Run test first
    print("Running test with mean-field case (d=0)...")
    test_results = run_experiment(cfg, test_mode=True, collect_diagnostics=args.diagnostics)
    print_test_summary(test_results, cfg)
    
    if args.test:
        print("\nTest mode complete.")
        save_results(test_results, cfg, filename="test_results.npz")
        plot_results(test_results, cfg)
        if args.diagnostics:
            plot_diagnostics(test_results, cfg, show_plots=False)
        return
    
    # Ask to continue
    if args.no_prompt:
        run_full = True
    else:
        response = input("\nRun full experiment? [y/N]: ").strip().lower()
        run_full = response == 'y'
    
    if run_full:
        print("\nRunning full experiment...")
        full_results = run_experiment(cfg, test_mode=False, collect_diagnostics=args.diagnostics)
        save_results(full_results, cfg)
        plot_results(full_results, cfg, show_plots=False)  # Don't block for full runs
        if args.diagnostics:
            plot_diagnostics(full_results, cfg, show_plots=False)
    else:
        print("Full experiment skipped. Test results available.")
        save_results(test_results, cfg, filename="test_results.npz")
        plot_results(test_results, cfg)
        if args.diagnostics:
            plot_diagnostics(test_results, cfg, show_plots=False)


if __name__ == "__main__":
    main()
