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
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from SSA.numba_sim_normal import make_normal_ssa_1d, SSANormalState

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Model parameters (equal kernels)
    b: float = 1.0               # Birth rate
    d_prime: float = 1.0         # Density-dependent death coefficient  
    sigma: float = 1.0           # Equal std for birth and death kernels
    
    # Domain
    L: float = 1000.0            # 1D periodic domain length (1000 particles at n=1)
    
    # d values to sweep
    d_controls: tuple = (0.0, 1e-4, 1e-3)
    d_test_range: tuple = (0.01, 0.1, 10)  # (start, stop, num_points)
    
    # Warmup
    initial_density_frac: float = 0.1   # Start at 10% of expected
    target_density_frac: float = 0.99   # Run until 99%
    extra_warmup_frac: float = 0.3      # Then run 30% more time
    
    # Measurement
    event_frac: float = 0.10     # Events per sample = 10% of equilibrium pop
    n_batches: int = 100         # Number of independent batches
    samples_per_batch: int = 100 # Samples per batch (more for autocorrelation handling)
    
    # PCF parameters
    dr: float = 0.1              # Bin width for g(r)
    r_max: float = 5.0           # Max r for g(r)
    
    # Parallel execution
    n_jobs: int = -1             # -1 = all cores
    
    # Output
    output_dir: str = "SSA/experiments/results"
    
    @property
    def total_samples(self) -> int:
        return self.n_batches * self.samples_per_batch
    
    def get_d_values(self) -> NDArray[np.float64]:
        """Get all d values to test."""
        d_test = np.linspace(*self.d_test_range)
        return np.concatenate([np.array(self.d_controls), d_test])
    
    def n_expected(self, d: float) -> float:
        """Expected mean-field density."""
        return (self.b - d) / self.d_prime if self.d_prime > 0 else 0.0
    
    def pop_expected(self, d: float) -> int:
        """Expected equilibrium population."""
        return int(self.n_expected(d) * self.L)


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
    # C(r) = FFT^{-1}(|FFT(rho)|^2) / N
    rho_hat = np.fft.fft(density)
    power = np.abs(rho_hat)**2
    corr = np.fft.ifft(power).real
    
    # Normalize: g(r) = C(r) / (N * n * dr)
    # where n = N/L is mean density
    n_density = N / L
    # corr[i] counts pairs at distance i*dr (both directions in 1D)
    # In 1D periodic, each distance bin has "width" dr
    # Expected pairs in bin from Poisson: N * n * dr (for each particle)
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
    
    # Compute autocorrelation
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        c = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean))
        autocorr[lag] = c / var
    
    # Integrated autocorrelation time: tau = 1 + 2 * sum(rho_k)
    # Cut off when autocorr drops below small threshold
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
    
    Returns dict with density statistics and g(r).
    """
    # Expected equilibrium
    n_exp = cfg.n_expected(d_val)
    pop_exp = cfg.pop_expected(d_val)
    
    if pop_exp < 10:
        # Too small, skip
        return {
            'd': d_val, 'n_expected': n_exp, 'pop_expected': pop_exp,
            'density_mean': np.nan, 'density_std': np.nan,
            'g_r': None, 'r_bins': None, 'warmup_time': 0.0,
            'measurement_time': 0.0, 'total_events': 0
        }
    
    # Create simulator
    sim = make_normal_ssa_1d(
        M=1,  # Single species
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
    
    # Start at 10% of expected density
    initial_pop = max(10, int(cfg.initial_density_frac * pop_exp))
    sim.spawn_random(0, initial_pop)
    
    # Run until 99% of expected
    target_pop = int(cfg.target_density_frac * pop_exp)
    t_sim_start = sim.current_time()
    
    while sim.current_population() < target_pop:
        sim.run_events(1000)
        if sim.current_population() <= 0:
            # Extinction - restart
            sim.spawn_random(0, initial_pop)
    
    t_99 = sim.current_time() - t_sim_start
    
    # Run 30% more time
    if t_99 > 0:
        sim.run_until_time(cfg.extra_warmup_frac * t_99)
    
    warmup_time = time.perf_counter() - warmup_start
    
    # === Measurement ===
    measure_start = time.perf_counter()
    
    densities = []
    g_r_accumulator = None
    r_bins = None
    
    n_events_per_sample = max(10, int(cfg.event_frac * pop_exp))
    total_events = 0
    
    for _ in range(cfg.total_samples):
        sim.run_events(n_events_per_sample)
        total_events += n_events_per_sample
        
        pop = sim.current_population()
        densities.append(pop / cfg.L)
        
        # Get positions for g(r)
        positions = np.array([sim.positions[i, 0] for i in range(pop)])
        r, gr = compute_pcf_fft_1d(positions, cfg.L, cfg.dr, cfg.r_max)
        
        if g_r_accumulator is None:
            r_bins = r
            g_r_accumulator = gr.copy()
        else:
            g_r_accumulator += gr
    
    measurement_time = time.perf_counter() - measure_start
    
    # Average g(r)
    g_r_mean = g_r_accumulator / cfg.total_samples
    
    # Compute density statistics with batch means
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
    
    Args:
        cfg: Experiment configuration
        test_mode: If True, only run mean-field case (d=0) for timing
    
    Returns dict with all results.
    """
    if test_mode:
        d_values = np.array([0.0])
        print("=== TEST MODE: Running mean-field case only ===")
    else:
        d_values = cfg.get_d_values()
    
    print(f"Domain: L={cfg.L}, σ={cfg.sigma}")
    print(f"Expected pop for d=0: {cfg.pop_expected(0)}")
    print(f"Samples per d: {cfg.total_samples} ({cfg.n_batches} batches × {cfg.samples_per_batch})")
    print(f"Events per sample: {int(cfg.event_frac * cfg.pop_expected(0))}")
    print(f"d values: {len(d_values)} cases")
    print()
    
    # Run in parallel
    results_list = Parallel(n_jobs=cfg.n_jobs if not test_mode else 1)(
        delayed(run_single_d)(d, cfg, seed=42 + i)
        for i, d in enumerate(tqdm(d_values, desc="Running simulations"))
    )
    
    # Organize results
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
        'r_bins': results_list[0]['r_bins'] if results_list[0]['r_bins'] is not None else None,
        'full_results': results_list,
    }
    
    return results


# =============================================================================
# Saving and Plotting
# =============================================================================
def save_results(results: dict, cfg: ExperimentConfig, filename: str = "kappa_results.npz"):
    """Save results to numpy file."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    g_r_stack = np.array([gr if gr is not None else np.full(len(results['r_bins']), np.nan) 
                         for gr in results['g_r_all']])
    
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
        L=cfg.L,
        b=cfg.b,
        d_prime=cfg.d_prime,
        sigma=cfg.sigma,
    )
    print(f"Results saved to {output_dir / filename}")


def plot_results(results: dict, cfg: ExperimentConfig, save_plots: bool = True):
    """Generate and optionally save plots."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    d_vals = results['d_values']
    n_meas = results['density_means']
    n_err = results['density_stds']
    n_exp = results['n_expected']
    
    # --- Plot 1: Density vs d ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.errorbar(d_vals, n_meas, yerr=n_err, fmt='o-', capsize=3, label='Measured')
    ax.plot(d_vals, n_exp, 'r--', label='Mean field: (b-d)/d\'', linewidth=2)
    ax.set_xlabel('d (intrinsic death rate)')
    ax.set_ylabel('Equilibrium density n')
    ax.set_title('First Moment: Density vs d')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot difference
    ax = axes[1]
    diff = n_meas - n_exp
    ax.errorbar(d_vals, diff, yerr=n_err, fmt='o-', capsize=3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('d (intrinsic death rate)')
    ax.set_ylabel('n_measured - n_expected')
    ax.set_title('Deviation from Mean Field')
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
        
        # Skip r=0 (first bin) as it has self-correlation artifact
        r_plot = r[1:]  # Exclude r=0
        
        # Use continuous colormap
        cmap = plt.cm.viridis
        d_min, d_max = d_vals.min(), d_vals.max()
        if d_max > d_min:
            norm = plt.Normalize(vmin=d_min, vmax=d_max)
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        
        for i, d in enumerate(d_vals):
            gr = results['g_r_all'][i]
            if gr is not None:
                color = cmap(norm(d))
                ax.plot(r_plot, gr[1:], color=color, alpha=0.8, linewidth=1.5)
        
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='Poisson')
        ax.set_xlabel('r')
        ax.set_ylabel('g(r)')
        ax.set_title('Pair Correlation Function')
        ax.set_xlim(r_plot[0], cfg.r_max)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('d (intrinsic death rate)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / 'pcf_g_r.png', dpi=150)
            print(f"Saved: {output_dir / 'pcf_g_r.png'}")
        plt.show()
    
    # --- Plot 3: Kappa extraction (linear fit) ---
    # n = (b-d)/d' - κ*d/d'
    # n - (b-d)/d' = -κ*d/d'
    # (n_exp - n_meas) * d' = κ * d
    if len(d_vals) > 3:
        mask = d_vals > 0.005  # Exclude d=0 and very small d
        if mask.sum() > 2:
            x = d_vals[mask]
            y = (n_exp[mask] - n_meas[mask]) * cfg.d_prime
            
            # Linear fit: y = κ * x
            kappa_est, _ = np.polyfit(x, y, 1, cov=False) if len(x) > 1 else (np.nan, 0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, s=50)
            x_fit = np.linspace(0, x.max() * 1.1, 100)
            ax.plot(x_fit, kappa_est * x_fit, 'r-', label=f'κ ≈ {kappa_est:.4f}')
            ax.set_xlabel('d')
            ax.set_ylabel('(n_expected - n_measured) × d\'')
            ax.set_title(f'Kappa Extraction: κ ≈ {kappa_est:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(output_dir / 'kappa_extraction.png', dpi=150)
                print(f"Saved: {output_dir / 'kappa_extraction.png'}")
            plt.show()


def print_test_summary(results: dict, cfg: ExperimentConfig):
    """Print summary for test run."""
    print("\n" + "="*60)
    print("TEST RUN SUMMARY (Mean Field Case: d=0)")
    print("="*60)
    
    r = results['full_results'][0]
    
    print(f"\nExpected density: {r['n_expected']:.4f}")
    print(f"Measured density: {r['density_mean']:.4f} ± {r['density_std']:.4f}")
    print(f"Relative error: {abs(r['density_mean'] - r['n_expected']) / r['n_expected'] * 100:.2f}%")
    
    print(f"\nWarmup time: {r['warmup_time']:.2f} s")
    print(f"Measurement time: {r['measurement_time']:.2f} s")
    print(f"Total time: {r['warmup_time'] + r['measurement_time']:.2f} s")
    
    print(f"\nTotal events: {r['total_events']:,}")
    print(f"Events/second: {r['total_events'] / r['measurement_time']:.0f}")
    
    # Autocorrelation estimate
    samples = r['density_samples']
    tau = estimate_autocorrelation_time(samples)
    ess = len(samples) / tau
    print(f"\nAutocorrelation time τ ≈ {tau:.1f}")
    print(f"Effective sample size: {ess:.0f} / {len(samples)}")
    
    # Extrapolate for full run
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
    cfg = ExperimentConfig()
    
    # Run test first
    print("Running test with mean-field case (d=0)...")
    test_results = run_experiment(cfg, test_mode=True)
    print_test_summary(test_results, cfg)
    
    # Ask to continue
    response = input("\nRun full experiment? [y/N]: ").strip().lower()
    if response == 'y':
        print("\nRunning full experiment...")
        full_results = run_experiment(cfg, test_mode=False)
        save_results(full_results, cfg)
        plot_results(full_results, cfg)
    else:
        print("Full experiment skipped. Test results available.")
        # Still save and plot test results
        save_results(test_results, cfg, filename="test_results.npz")
        plot_results(test_results, cfg)


if __name__ == "__main__":
    main()
