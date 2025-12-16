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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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
    rho_hat = np.fft.fft(density)
    power = np.abs(rho_hat)**2
    corr = np.fft.ifft(power).real
    
    # Normalize: g(r) = C(r) / (N * n * dr)
    n_density = N / L
    g_r = corr / (N * n_density * dr)
    
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
                 progress_callback: Callable | None = None) -> dict:
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
            'warmup_time': 0.0, 'measurement_time': 0.0, 'total_events': 0
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
    
    # === Warmup ===
    warmup_start = time.perf_counter()
    
    initial_pop = max(10, int(cfg.initial_density_frac * pop_exp))
    sim.spawn_random(0, initial_pop)
    
    target_pop = int(cfg.target_density_frac * pop_exp)
    t_sim_start = sim.current_time()
    
    while sim.current_population() < target_pop:
        sim.run_events(1000)
        if sim.current_population() <= 0:
            sim.spawn_random(0, initial_pop)
    
    t_99 = sim.current_time() - t_sim_start
    
    if t_99 > 0:
        sim.run_until_time(cfg.extra_warmup_frac * t_99)
    
    warmup_time = time.perf_counter() - warmup_start
    
    # === Measurement ===
    measure_start = time.perf_counter()
    
    densities = []
    g_r_samples = []  # Store all g(r) samples for batch statistics
    r_bins = None
    
    n_events_per_sample = max(10, int(cfg.event_frac * pop_exp))
    total_events = 0
    
    for _ in range(cfg.total_samples):
        sim.run_events(n_events_per_sample)
        total_events += n_events_per_sample
        
        pop = sim.current_population()
        densities.append(pop / cfg.L)
        
        positions = np.array([sim.positions[i, 0] for i in range(pop)])
        r, gr = compute_pcf_fft_1d(positions, cfg.L, cfg.dr, cfg.r_max)
        
        if r_bins is None:
            r_bins = r
        g_r_samples.append(gr)
    
    measurement_time = time.perf_counter() - measure_start
    
    # Convert to arrays
    g_r_samples = np.array(g_r_samples)  # [n_samples, n_r_bins]
    
    # Average g(r)
    g_r_mean = g_r_samples.mean(axis=0)
    
    # Compute batch means for g(r) at each r bin
    batch_size = cfg.total_samples // cfg.n_batches
    g_r_batches = np.array([
        g_r_samples[i*batch_size:(i+1)*batch_size].mean(axis=0)
        for i in range(cfg.n_batches)
    ])  # [n_batches, n_r_bins]
    
    # Compute density statistics
    density_mean, density_std, batch_means = compute_batch_statistics(
        np.array(densities), cfg.n_batches
    )
    
    return {
        'd': d_val,
        'n_expected': n_exp,
        'pop_expected': pop_exp,
        'density_mean': density_mean,
        'density_std': density_std,
        'density_samples': np.array(densities),
        'batch_means': batch_means,
        'g_r': g_r_mean,
        'g_r_batches': g_r_batches,
        'r_bins': r_bins,
        'warmup_time': warmup_time,
        'measurement_time': measurement_time,
        'total_events': total_events,
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================
def run_experiment(cfg: ExperimentConfig, test_mode: bool = False) -> dict:
    """
    Run full experiment across all d values.
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
    print()
    
    results_list = Parallel(n_jobs=cfg.n_jobs if not test_mode else 1)(
        delayed(run_single_d)(d, cfg, seed=42 + i)
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


def plot_results(results: dict, cfg: ExperimentConfig, save_plots: bool = True):
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
    plt.show()
    
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
        plt.show()
    
    # --- Plot 3: PCF values at specific r vs d with confidence bands ---
    plot_pcf_vs_d(results, cfg, save_plots)
    
    # --- Plot 4: Kappa extraction (linear fit) ---
    if len(d_vals) > 3:
        mask = d_vals > 0.005
        if mask.sum() > 2:
            x = d_vals[mask]
            y = (n_exp[mask] - n_meas[mask]) * cfg.d_prime
            
            kappa_est, _ = np.polyfit(x, y, 1, cov=False) if len(x) > 1 else (np.nan, 0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, s=50)
            x_fit = np.linspace(0, x.max() * 1.1, 100)
            ax.plot(x_fit, kappa_est * x_fit, 'r-', label=f'κ ≈ {kappa_est:.4f}')
            ax.set_xlabel('d')
            ax.set_ylabel('(n_expected - n_measured) × d\'')
            ax.set_title(f'Kappa Extraction: κ ≈ {kappa_est:.4f} (σ={cfg.sigma})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(output_dir / 'kappa_extraction.png', dpi=150)
                print(f"Saved: {output_dir / 'kappa_extraction.png'}")
            plt.show()


def plot_pcf_vs_d(results: dict, cfg: ExperimentConfig, save_plots: bool = True):
    """
    Plot PCF values at first 6 r values in a 3x2 grid vs d 
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
    plt.show()


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
    
    samples = r['density_samples']
    tau = estimate_autocorrelation_time(samples)
    ess = len(samples) / tau
    print(f"\nAutocorrelation time τ ≈ {tau:.1f}")
    print(f"Effective sample size: {ess:.0f} / {len(samples)}")
    
    n_d_full = len(cfg.get_d_values())
    time_per_d = r['warmup_time'] + r['measurement_time']
    print(f"\n--- Extrapolation for full run ({n_d_full} d-values) ---")
    print(f"Estimated total time (serial): {time_per_d * n_d_full / 60:.1f} min")
    
    n_cores = os.cpu_count() or 4
    print(f"Estimated total time ({n_cores} cores): {time_per_d * n_d_full / n_cores / 60:.1f} min")


# =============================================================================
# Entry Point
# =============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Kappa Estimation Experiment")
    parser.add_argument('--config', '-c', type=str, default='baseline',
                       choices=list_configs(),
                       help=f"Configuration to use. Available: {list_configs()}")
    parser.add_argument('--test', '-t', action='store_true',
                       help="Run test mode only (d=0)")
    parser.add_argument('--no-prompt', '-y', action='store_true',
                       help="Skip confirmation prompt, run full experiment")
    
    args = parser.parse_args()
    
    cfg = get_config(args.config)
    print(f"Using configuration: {cfg.name}")
    print(f"  sigma={cfg.sigma}, dr={cfg.dr}, r_max={cfg.r_max}")
    print()
    
    # Run test first
    print("Running test with mean-field case (d=0)...")
    test_results = run_experiment(cfg, test_mode=True)
    print_test_summary(test_results, cfg)
    
    if args.test:
        print("\nTest mode complete.")
        save_results(test_results, cfg, filename="test_results.npz")
        plot_results(test_results, cfg)
        return
    
    # Ask to continue
    if args.no_prompt:
        run_full = True
    else:
        response = input("\nRun full experiment? [y/N]: ").strip().lower()
        run_full = response == 'y'
    
    if run_full:
        print("\nRunning full experiment...")
        full_results = run_experiment(cfg, test_mode=False)
        save_results(full_results, cfg)
        plot_results(full_results, cfg)
    else:
        print("Full experiment skipped. Test results available.")
        save_results(test_results, cfg, filename="test_results.npz")
        plot_results(test_results, cfg)


if __name__ == "__main__":
    main()
