"""
Extinction Limit Scaling - Refactored

Measures equilibrium density across (d, d') parameter space using dual-simulation
coupling method. Two-phase approach:
1. Calibration: Run until convergence to measure equilibrium density
2. Measurement: Rescale area for target population (~1000), run final measurement

Results saved incrementally to CSV. Analysis performed after all simulations complete.
"""
from __future__ import annotations
import sys
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Iterator
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from SSA.numba_sim_normal import make_normal_ssa_1d, get_all_particle_coords

# Physical constants
B, SIGMA = 1.0, 1.0

# Default d' values to simulate
DPRIME_VALUES = [1.0, 0.1, 0.01]

# Fine grid ranges centered on expected extinction thresholds
FINE_GRID_RANGES = {
    1.0: (0.1, 0.3),    # d' = 1.0: extinction near d ≈ 0.2
    0.1: (0.6, 0.8),    # d' = 0.1: extinction near d ≈ 0.7
    0.01: (0.8, 1.0),   # d' = 0.01: extinction near d ≈ 0.9
}


@dataclass
class SimulationParams:
    """Parameters for a single simulation point."""
    d: float
    d_prime: float
    is_fine_grid: bool
    
    def to_tuple(self) -> Tuple[float, float, bool]:
        return (self.d, self.d_prime, self.is_fine_grid)


@dataclass
class SimulationResult:
    """Results from a single simulation."""
    sim_id: int
    seed: int
    d: float
    d_prime: float
    is_fine_grid: bool
    density_eq: float
    density_se: float
    xi: float
    t_eq: Optional[int]
    L: float
    converged: bool
    n_eff: float
    n_samples: int
    calibration_density: float
    calibration_L: float
    z_mean: float
    pcf_pval: float


def create_parameter_grid(
    dprime_values: List[float] = None,
    d_coarse_step: float = 0.1,
    d_fine_step: float = 0.01
) -> List[SimulationParams]:
    """
    Create parameter grid as list of (d, d') tuples.
    
    For each d' value:
    - Coarse grid: d ∈ [0, 0.9] with step 0.1 (excluding fine region)
    - Fine grid: centered on expected extinction threshold with step 0.01
    
    Returns:
        List of SimulationParams, each representing one (d, d') point.
    """
    if dprime_values is None:
        dprime_values = DPRIME_VALUES
    
    params_list = []
    
    for d_prime in dprime_values:
        # Get fine grid range for this d'
        fine_start, fine_end = FINE_GRID_RANGES.get(d_prime, (0.9, 1.0))
        
        # Generate coarse grid (excluding fine region)
        d_coarse = [d for d in np.arange(0, 0.91, d_coarse_step) 
                    if d < fine_start - 0.05 or d > fine_end + 0.05]
        
        # Generate fine grid
        d_fine = list(np.arange(fine_start, fine_end + d_fine_step/2, d_fine_step))
        
        # Combine and create params
        for d in d_coarse:
            params_list.append(SimulationParams(d=float(d), d_prime=d_prime, is_fine_grid=False))
        for d in d_fine:
            params_list.append(SimulationParams(d=float(d), d_prime=d_prime, is_fine_grid=True))
    
    # Sort by (d_prime, d) for consistent ordering
    params_list.sort(key=lambda p: (p.d_prime, p.d))
    return params_list


def get_fine_grid_bounds(d_prime: float) -> Tuple[float, float]:
    """Get fine grid bounds for constraining extinction threshold fit."""
    return FINE_GRID_RANGES.get(d_prime, (0.9, 1.0))


# =============================================================================
# CSV Results Writer - Incremental saving with buffering
# =============================================================================
class CSVResultsWriter:
    """
    Incremental CSV writer for simulation results.
    
    Writes results row-by-row with periodic flushing to disk.
    Thread-safe for parallel execution via file locking.
    """
    
    FIELDNAMES = [
        'sim_id', 'seed', 'd', 'd_prime', 'is_fine_grid', 'density_eq', 'density_se', 
        'xi', 't_eq', 'L', 'converged', 'n_eff', 'n_samples',
        'calibration_density', 'calibration_L', 'z_mean', 'pcf_pval'
    ]
    
    def __init__(self, filepath: Path, buffer_size: int = 5):
        """
        Initialize CSV writer.
        
        Args:
            filepath: Path to CSV file
            buffer_size: Number of results to buffer before flushing
        """
        self.filepath = Path(filepath)
        self.buffer_size = buffer_size
        self.buffer: List[SimulationResult] = []
        self._file_initialized = False
        
    def _init_file(self):
        """Create CSV file with header if it doesn't exist."""
        if not self._file_initialized:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            if not self.filepath.exists():
                with open(self.filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()
            self._file_initialized = True
    
    def write_result(self, result: SimulationResult):
        """Add result to buffer, flush if buffer is full."""
        self.buffer.append(result)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered results to disk."""
        if not self.buffer:
            return
        self._init_file()
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            for result in self.buffer:
                row = asdict(result)
                # Convert None to empty string for CSV
                row = {k: ('' if v is None else v) for k, v in row.items()}
                writer.writerow(row)
        self.buffer.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


def load_results_csv(filepath: Path) -> List[SimulationResult]:
    """Load results from CSV file."""
    results = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Handle legacy CSVs without sim_id/seed columns
            sim_id = int(row.get('sim_id', idx)) if row.get('sim_id') else idx
            seed = int(row.get('seed', 42 + idx * 100)) if row.get('seed') else 42 + idx * 100
            results.append(SimulationResult(
                sim_id=sim_id,
                seed=seed,
                d=float(row['d']),
                d_prime=float(row['d_prime']),
                is_fine_grid=row['is_fine_grid'].lower() == 'true',
                density_eq=float(row['density_eq']) if row['density_eq'] else 0.0,
                density_se=float(row['density_se']) if row['density_se'] else np.nan,
                xi=float(row['xi']) if row['xi'] else np.nan,
                t_eq=int(row['t_eq']) if row['t_eq'] else None,
                L=float(row['L']) if row['L'] else np.nan,
                converged=row['converged'].lower() == 'true',
                n_eff=float(row['n_eff']) if row['n_eff'] else 0.0,
                n_samples=int(row['n_samples']) if row['n_samples'] else 0,
                calibration_density=float(row['calibration_density']) if row['calibration_density'] else 0.0,
                calibration_L=float(row['calibration_L']) if row['calibration_L'] else 0.0,
                z_mean=float(row['z_mean']) if row['z_mean'] else np.nan,
                pcf_pval=float(row['pcf_pval']) if row['pcf_pval'] else np.nan,
            ))
    return results


def fit_power_law_extinction(d_vals: NDArray, density: NDArray, density_se: NDArray,
                              fine_mask: NDArray) -> dict:
    """
    Fit power-law model: density = A * (d_ext - d)^β
    
    Uses only fine grid data where density > 0.
    
    Parameters:
        d_vals: array of d values
        density: measured density values
        density_se: standard error of density measurements
        fine_mask: boolean mask for fine grid points
    
    Returns dict with:
        d_ext: fitted extinction threshold
        beta: fitted exponent
        A: fitted amplitude
        r_squared: goodness of fit
        fit_d: d values used for fit
        fit_density: density values used for fit
    """
    # Select fine grid points with positive density
    mask = fine_mask & (density > 0) & np.isfinite(density)
    if np.sum(mask) < 3:
        return {'d_ext': np.nan, 'beta': np.nan, 'A': np.nan, 'r_squared': np.nan,
                'fit_d': np.array([]), 'fit_density': np.array([])}
    
    d_fit = d_vals[mask]
    n_fit = density[mask]
    se_fit = density_se[mask]
    
    # Replace NaN standard errors with mean of valid ones
    se_valid = se_fit[np.isfinite(se_fit)]
    if len(se_valid) > 0:
        se_fit = np.where(np.isfinite(se_fit), se_fit, np.mean(se_valid))
    else:
        se_fit = np.ones_like(se_fit) * 0.01 * np.mean(n_fit)
    
    # Initial guess: d_ext slightly above max d with positive density
    d_ext_guess = np.max(d_fit) + 0.05
    
    def power_law(d, A, beta, d_ext):
        """Power law model: n = A * (d_ext - d)^beta"""
        return A * np.maximum(d_ext - d, 1e-10)**beta
    
    try:
        # Bounds: A > 0, beta > 0, d_ext > max(d_fit)
        popt, pcov = curve_fit(
            power_law, d_fit, n_fit,
            p0=[n_fit[0], 1.0, d_ext_guess],
            bounds=([0.001, 0.1, np.max(d_fit) + 0.001], [100, 5.0, 1.5]),
            sigma=se_fit,
            absolute_sigma=True,
            maxfev=5000
        )
        A, beta, d_ext = popt
        
        # Compute R²
        n_pred = power_law(d_fit, A, beta, d_ext)
        ss_res = np.sum((n_fit - n_pred)**2)
        ss_tot = np.sum((n_fit - np.mean(n_fit))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return {
            'd_ext': d_ext, 'beta': beta, 'A': A, 'r_squared': r_squared,
            'fit_d': d_fit, 'fit_density': n_fit, 'fit_se': se_fit
        }
    except (RuntimeError, ValueError) as e:
        print(f"Power-law fit failed: {e}")
        return {'d_ext': np.nan, 'beta': np.nan, 'A': np.nan, 'r_squared': np.nan,
                'fit_d': d_fit, 'fit_density': n_fit}

def mean_field_density(d: float, d_prime: float) -> float:
    return max(0.0, (B - d) / d_prime)

def estimate_autocorr_time(samples: NDArray, max_lag: int = 50) -> float:
    n = len(samples)
    if n < 10: return 1.0
    max_lag = min(max_lag, n // 4)
    mean, var = samples.mean(), samples.var()
    if var < 1e-12: return 1.0
    tau = 1.0
    for k in range(1, max_lag):
        rho_k = np.mean((samples[:-k] - mean) * (samples[k:] - mean)) / var
        if rho_k < 0.05: break
        tau += 2 * rho_k
    return max(1.0, tau)

def compute_pcf_1d(positions: NDArray, L: float, n_bins: int, r_max: float) -> tuple[NDArray, NDArray]:
    if len(positions) < 2:
        return np.linspace(0, r_max, n_bins), np.ones(n_bins)
    pos = positions.flatten()
    n = len(pos)
    # Vectorized pairwise distances (periodic)
    diff = np.abs(pos[:, None] - pos[None, :])
    diff = np.minimum(diff, L - diff)
    upper_tri = np.triu_indices(n, k=1)
    dists = diff[upper_tri]
    dists = dists[dists <= r_max]
    if len(dists) == 0:
        return np.linspace(0, r_max, n_bins), np.ones(n_bins)
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    counts, _ = np.histogram(dists, bins=r_edges)
    expected = n * (n - 1) / 2 * (2 * dr / L)
    return r_centers, counts / max(expected, 1e-10)

def compute_integral_correlation_length(r: NDArray, g_r: NDArray, L_half: float) -> float:
    """Integral correlation length: ∫|g(r)-1|dr, limited by L/2."""
    mask = (r > 0) & (r <= L_half)
    if np.sum(mask) < 3:
        return SIGMA
    return max(SIGMA, np.trapz(np.abs(g_r[mask] - 1.0), r[mask]))


def _exp_decay(r: NDArray, A: float, xi: float) -> NDArray:
    """Exponential decay model: g(r) - 1 = A * exp(-r/xi)."""
    return A * np.exp(-r / xi)


def compute_exponential_correlation_length(r: NDArray, g_r: NDArray, L_half: float) -> float:
    """
    Compute correlation length by fitting g(r) - 1 = A * exp(-r/xi).
    
    Falls back to integral method if fit fails.
    """
    mask = (r > 0) & (r <= L_half)
    if np.sum(mask) < 5:
        return SIGMA
    
    r_fit = r[mask]
    h_fit = g_r[mask] - 1.0  # h(r) = g(r) - 1
    
    # Initial guess: A from max deviation, xi from SIGMA
    A0 = h_fit[0] if len(h_fit) > 0 else 1.0
    if np.abs(A0) < 1e-10:
        A0 = np.max(np.abs(h_fit)) if len(h_fit) > 0 else 1.0
    xi0 = SIGMA
    
    try:
        # Fit exponential decay to |h(r)| with sign of A matching h(r) sign
        sign = 1.0 if np.mean(h_fit[:min(5, len(h_fit))]) >= 0 else -1.0
        popt, _ = curve_fit(
            _exp_decay, 
            r_fit, 
            sign * h_fit,  # Make positive for fitting
            p0=[np.abs(A0), xi0],
            bounds=([0, SIGMA * 0.1], [100, L_half]),
            maxfev=1000
        )
        xi = popt[1]
        return max(SIGMA, xi)
    except (RuntimeError, ValueError):
        # Fall back to integral method
        return compute_integral_correlation_length(r, g_r, L_half)

def pcf_permutation_test(pos1: NDArray, pos2: NDArray, L: float, n_bins: int = 50, n_perm: int = 100):
    all_pos = np.concatenate([pos1.flatten(), pos2.flatten()])
    n1, n2 = len(pos1.flatten()), len(pos2.flatten())
    L_half = L / 2.0
    if n1 < 10 or n2 < 10:
        return 0.0, 1.0, SIGMA, None, None, None
    
    r_max_init = min(10 * SIGMA, L_half)
    r1, g1 = compute_pcf_1d(pos1, L, n_bins, r_max_init)
    r2, g2 = compute_pcf_1d(pos2, L, n_bins, r_max_init)
    # Use exponential fit for correlation length
    xi_avg = 0.5 * (compute_exponential_correlation_length(r1, g1, L_half) +
                    compute_exponential_correlation_length(r2, g2, L_half))
    
    r_max = min(6 * xi_avg, L_half)
    r1, g1 = compute_pcf_1d(pos1, L, n_bins, r_max)
    r2, g2 = compute_pcf_1d(pos2, L, n_bins, r_max)
    
    mask = (r1 >= 0.1 * xi_avg) & (r1 <= min(5 * xi_avg, L_half))
    dr = r1[1] - r1[0] if len(r1) > 1 else 1.0
    obs_stat = np.sum((g1[mask] - g2[mask])**2) * dr if np.sum(mask) >= 2 else 0.0
    
    perm_stats = []
    for _ in range(n_perm):
        perm = np.random.permutation(len(all_pos))
        _, pg1 = compute_pcf_1d(all_pos[perm[:n1]].reshape(-1, 1), L, n_bins, r_max)
        _, pg2 = compute_pcf_1d(all_pos[perm[n1:]].reshape(-1, 1), L, n_bins, r_max)
        perm_stats.append(np.sum((pg1[mask] - pg2[mask])**2) * dr if np.sum(mask) >= 2 else 0.0)
    
    return obs_stat, np.mean(np.array(perm_stats) >= obs_stat), xi_avg, r1, g1, g2

def detect_equilibrium_dual_sim(
    d: float, d_prime: float, L: float, N_lo_start: int = 100, N_up_start: int = 1200,
    max_char_times: int = 5000, min_window: int = 10, pcf_sig: float = 0.05,
    target_rel_error: float = 0.01, seed: int = 42, verbose: bool = False,
    target_eq_population: int = 1000, diagnostic_writer: Optional[DiagnosticWriter] = None
) -> dict:
    """
    Detect equilibrium using dual-simulation coupling method.
    
    Equilibrium Detection Theory:
    =============================
    Two independent simulations are started from different initial conditions
    (low and high population). If they converge to the same statistical state,
    the system has reached equilibrium. This "coupling from the past" approach
    is theoretically sound because:
    
    1. ERGODICITY: For ergodic Markov processes, the stationary distribution
       is unique and independent of initial conditions.
    
    2. CROSSING CRITERION: When the low-start trajectory crosses above the
       high-start trajectory, both processes have "mixed" sufficiently that
       their distributions should overlap. This is a necessary (not sufficient)
       condition for equilibration.
    
    3. STATISTICAL TESTS: After crossing, we verify equilibrium via:
       - Mean comparison: z-test with |z| < 2.0 (roughly p > 0.05)
       - PCF comparison: permutation test ensuring spatial structure matches
    
    4. BURN-IN: We wait 2*min_window steps after crossing before testing,
       allowing autocorrelation to decay.
    
    Area Scaling Strategy:
    ======================
    1. If extinction occurs before equilibrium, increase area by 10x and retry.
    2. After 1st convergence, rescale area so that equilibrium population ≈ target_eq_population
       (default 1000) based on observed 1st convergence density. This ensures
       accurate measurements with sufficient particles.
    
    Error Estimation:
    =================
    Post-equilibrium sampling with autocorrelation correction:
    - Effective sample size: n_eff = n / τ (τ = autocorrelation time)
    - SE = σ / √n_eff
    - Target: SE/mean < target_rel_error/1.96 for 1% error at 95% CI
    
    Returns dict with convergence info, density/PCF estimates with SE.
    """
    area_increases, current_L = [], L
    extinct_traces = []  # Store traces from simulations that went extinct
    first_convergence = None  # Store 1st convergence info for area rescaling
    
    for attempt in range(5):  # Increased from 3 to allow for area rescaling attempt
        # Limit cells to prevent memory issues (max 100, min 10)
        n_cells = max(10, min(100, int(current_L / (5 * SIGMA))))
        sim_lo = make_normal_ssa_1d(M=1, area_len=current_L, birth_rates=[B], death_rates=[d],
            dd_matrix=[[d_prime]], birth_std=[SIGMA], death_std=[[SIGMA]],
            death_cull_sigmas=5.0, is_periodic=True, seed=seed, cell_count=n_cells)
        sim_up = make_normal_ssa_1d(M=1, area_len=current_L, birth_rates=[B], death_rates=[d],
            dd_matrix=[[d_prime]], birth_std=[SIGMA], death_std=[[SIGMA]],
            death_cull_sigmas=5.0, is_periodic=True, seed=seed + 1000, cell_count=n_cells)
        
        sim_lo.spawn_random(0, N_lo_start)
        sim_up.spawn_random(0, N_up_start)
        
        trace_lo, trace_up = [], []
        crossed, t_cross = False, None
        
        for t in range(max_char_times):
            N_lo, N_up = sim_lo.current_population(), sim_up.current_population()
            trace_lo.append(N_lo); trace_up.append(N_up)
            
            # Write trace data if diagnostic writer is provided
            if diagnostic_writer is not None:
                stage = "equilibrium" if crossed else "warmup"
                diagnostic_writer.write_trace_row(t, N_lo, N_up, stage)
            
            if N_lo == 0 or N_up == 0:
                if verbose: print(f"  Extinction at t={t}, L={current_L:.0f}")
                # Store extinct trace before area increase
                extinct_traces.append({
                    'L': current_L, 't': t,
                    'trace_lo': np.array(trace_lo),
                    'trace_up': np.array(trace_up)
                })
                area_increases.append({'L': current_L, 't': t})
                current_L *= 10
                break
            
            if N_lo > N_up and not crossed:
                crossed, t_cross = True, t
            
            if crossed and t >= t_cross + 2 * min_window:
                tau = max(estimate_autocorr_time(np.array(trace_lo[-2*min_window:])),
                          estimate_autocorr_time(np.array(trace_up[-2*min_window:])))
                window = max(min_window, int(np.ceil(tau)))
                
                if len(trace_lo) >= 2 * window:
                    mean_lo, mean_up = np.mean(trace_lo[-window:]), np.mean(trace_up[-window:])
                    var_lo, var_up = np.var(trace_lo[-window:]), np.var(trace_up[-window:])
                    se = np.sqrt(var_lo/window + var_up/window + 1e-10)
                    z_mean = abs(mean_lo - mean_up) / se
                    
                    pos_lo, pos_up = get_all_particle_coords(sim_lo)[0], get_all_particle_coords(sim_up)[0]
                    _, pcf_pval, xi_avg, r, g_lo, g_up = pcf_permutation_test(pos_lo, pos_up, current_L)
                    
                    if z_mean < 2.0 and pcf_pval > pcf_sig:
                        # 1st convergence detected - check if area rescaling needed
                        first_eq_density = 0.5 * (mean_lo + mean_up) / current_L
                        current_eq_pop = first_eq_density * current_L
                        
                        # Calculate new L to achieve target_eq_population
                        if first_eq_density > 1e-10:
                            new_L = target_eq_population / first_eq_density
                        else:
                            new_L = current_L
                        
                        # If new_L is significantly larger (>1.5x) and this is 1st convergence, rescale
                        if first_convergence is None and new_L > current_L * 1.5:
                            if verbose:
                                print(f"  1st convergence: density={first_eq_density:.4f}, pop={current_eq_pop:.0f}")
                                print(f"  Rescaling area: L={current_L:.0f} -> {new_L:.0f} for target pop ~{target_eq_population}")
                            
                            # Store 1st convergence info
                            first_convergence = {
                                'density': first_eq_density,
                                'L': current_L,
                                't_eq': t,
                                't_cross': t_cross,
                                'xi': xi_avg,
                                'trace_lo': np.array(trace_lo),
                                'trace_up': np.array(trace_up)
                            }
                            
                            # Update area and initial populations for 2nd convergence run
                            area_increases.append({'L': current_L, 't': t, 'reason': 'rescale_to_target'})
                            current_L = new_L
                            N_lo_start = max(10, int(0.1 * target_eq_population))
                            N_up_start = int(1.2 * target_eq_population)
                            break  # Exit inner loop to restart with new area
                        
                        # Proceed with post-equilibrium sampling (final measurements)
                        density_samples = []
                        pcf_samples = []  # Time series of PCF samples
                        eq_positions_snapshots = []  # Store position snapshots for diagnostics
                        L_half = current_L / 2.0
                        r_max_pcf = min(6 * xi_avg, L_half)
                        pcf_running = PCFRunningStats(50) if diagnostic_writer is not None else None
                        
                        max_samples = min(500, int((1.96 / target_rel_error)**2 * 5))
                        
                        for sample_idx in range(max_samples):
                            sim_lo.run_events(max(10, N_lo // 2))
                            sim_up.run_events(max(10, N_up // 2))
                            N_lo, N_up = sim_lo.current_population(), sim_up.current_population()
                            if N_lo == 0 or N_up == 0:
                                break
                            
                            current_density = (N_lo + N_up) / 2 / current_L
                            density_samples.append(current_density)
                            
                            # Write density sample to diagnostic file
                            if diagnostic_writer is not None:
                                diagnostic_writer.write_density_sample(sample_idx, current_density, N_lo, N_up)
                            
                            # Collect PCF samples every 20 steps
                            if len(density_samples) % 20 == 0:
                                pos_lo = get_all_particle_coords(sim_lo)[0]
                                pos_up = get_all_particle_coords(sim_up)[0]
                                r_bins, g1 = compute_pcf_1d(pos_lo, current_L, 50, r_max_pcf)
                                _, g2 = compute_pcf_1d(pos_up, current_L, 50, r_max_pcf)
                                g_avg = 0.5 * (g1 + g2)
                                pcf_samples.append(g_avg)
                                
                                # Update running PCF statistics and write to file
                                if pcf_running is not None:
                                    pcf_running.update(r_bins, g_avg)
                                    g_mean, g_se = pcf_running.get_mean_and_se()
                                    diagnostic_writer.write_pcf_sample(
                                        sample_idx // 20, r_bins, g_mean, g_se, N_lo + N_up
                                    )
                                
                                # Store position snapshot for diagnostics (every 100 samples)
                                if sample_idx % 100 == 0:
                                    eq_positions_snapshots.append(np.concatenate([pos_lo.flatten(), pos_up.flatten()]))
                            
                            if len(density_samples) >= 50 and len(density_samples) % 50 == 0:
                                samples = np.array(density_samples)
                                tau_new = estimate_autocorr_time(samples)
                                n_eff = len(samples) / max(tau_new, 1.0)
                                se_rel = np.std(samples) / (np.mean(samples) * np.sqrt(n_eff) + 1e-10)
                                if se_rel < target_rel_error / 1.96:
                                    break
                        
                        samples = np.array(density_samples)
                        tau_final = estimate_autocorr_time(samples)
                        n_eff = len(samples) / tau_final
                        mean_density = np.mean(samples)
                        se_density = np.std(samples) / np.sqrt(n_eff)
                        
                        # Compute PCF mean and SE from time series
                        if len(pcf_samples) > 0:
                            pcf_arr = np.array(pcf_samples)
                            pcf_mean = np.mean(pcf_arr, axis=0)
                            pcf_se = np.std(pcf_arr, axis=0) / np.sqrt(len(pcf_arr))
                        else:
                            pcf_mean = 0.5 * (g_lo + g_up)
                            pcf_se = np.zeros_like(pcf_mean)
                        
                        # Use exponential fit for final correlation length
                        xi_final = compute_exponential_correlation_length(r, pcf_mean, L_half) if r is not None else xi_avg
                        
                        # Get final position snapshot for diagnostics
                        final_pos_lo = get_all_particle_coords(sim_lo)[0]
                        final_pos_up = get_all_particle_coords(sim_up)[0]
                        eq_positions = np.concatenate([final_pos_lo.flatten(), final_pos_up.flatten()])
                        
                        return {
                            'converged': True, 'L': current_L, 't_eq': t, 't_cross': t_cross,
                            'density_eq': mean_density, 'density_se': se_density,
                            'density_ci95': 1.96 * se_density, 'n_eff': n_eff,
                            'n_samples': len(samples), 'tau_final': tau_final,
                            'trace_lo': np.array(trace_lo), 'trace_up': np.array(trace_up),
                            'area_increases': area_increases, 'extinct_traces': extinct_traces,
                            'first_convergence': first_convergence,
                            'xi': xi_final, 'r_pcf': r,
                            'pcf_mean': pcf_mean, 'pcf_se': pcf_se, 'z_mean': z_mean, 'pcf_pval': pcf_pval,
                            'eq_positions': eq_positions, 'eq_positions_snapshots': eq_positions_snapshots
                        }
            
            sim_lo.run_events(max(10, (N_lo + N_up) // 2))
            sim_up.run_events(max(10, (N_lo + N_up) // 2))
        else:
            mean_val = 0.5 * (np.mean(trace_lo[-min_window:]) + np.mean(trace_up[-min_window:]))
            return {'converged': False, 'L': current_L, 't_eq': max_char_times, 't_cross': t_cross,
                    'density_eq': mean_val / current_L, 'density_se': np.nan, 'density_ci95': np.nan,
                    'n_eff': 0, 'n_samples': 0, 'tau_final': np.nan,
                    'trace_lo': np.array(trace_lo), 'trace_up': np.array(trace_up),
                    'area_increases': area_increases, 'extinct_traces': extinct_traces,
                    'first_convergence': first_convergence,
                    'xi': SIGMA, 'r_pcf': None, 'pcf_mean': None,
                    'z_mean': np.nan, 'pcf_pval': np.nan,
                    'eq_positions': None, 'eq_positions_snapshots': []}
    
    return {'converged': False, 'L': current_L, 't_eq': None, 't_cross': None,
            'density_eq': 0.0, 'density_se': np.nan, 'density_ci95': np.nan,
            'n_eff': 0, 'n_samples': 0, 'tau_final': np.nan,
            'trace_lo': np.array([]), 'trace_up': np.array([]),
            'area_increases': area_increases, 'extinct_traces': extinct_traces,
            'first_convergence': first_convergence,
            'xi': SIGMA, 'r_pcf': None, 'pcf_mean': None,
            'z_mean': np.nan, 'pcf_pval': np.nan,
            'eq_positions': None, 'eq_positions_snapshots': []}

# =============================================================================
# Two-Phase Simulation: Calibration + Measurement
# =============================================================================
def run_calibration_phase(
    d: float, d_prime: float, 
    initial_L: float,
    max_char_times: int = 3000,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[float, float, bool]:
    """
    Phase 1: Run simulation until convergence to measure equilibrium density.
    
    Uses smaller population for faster calibration.
    
    Returns:
        (calibration_density, calibration_L, converged)
    """
    if d >= B:
        return 0.0, initial_L, False
    
    current_L = initial_L
    calibration_pop = 200  # Use smaller population for calibration
    
    for attempt in range(3):
        n_cells = max(10, min(50, int(current_L / (5 * SIGMA))))
        sim_lo = make_normal_ssa_1d(M=1, area_len=current_L, birth_rates=[B], death_rates=[d],
            dd_matrix=[[d_prime]], birth_std=[SIGMA], death_std=[[SIGMA]],
            death_cull_sigmas=5.0, is_periodic=True, seed=seed, cell_count=n_cells)
        sim_up = make_normal_ssa_1d(M=1, area_len=current_L, birth_rates=[B], death_rates=[d],
            dd_matrix=[[d_prime]], birth_std=[SIGMA], death_std=[[SIGMA]],
            death_cull_sigmas=5.0, is_periodic=True, seed=seed + 500, cell_count=n_cells)
        
        N_lo_start = max(10, int(0.1 * calibration_pop))
        N_up_start = int(1.2 * calibration_pop)
        sim_lo.spawn_random(0, N_lo_start)
        sim_up.spawn_random(0, N_up_start)
        
        crossed, t_cross = False, None
        min_window = 10
        
        for t in range(max_char_times):
            N_lo, N_up = sim_lo.current_population(), sim_up.current_population()
            
            if N_lo == 0 or N_up == 0:
                if verbose: print(f"  Calibration extinction at t={t}, L={current_L:.0f}")
                current_L *= 10
                break
            
            if N_lo > N_up and not crossed:
                crossed, t_cross = True, t
            
            # Quick convergence check
            if crossed and t >= t_cross + min_window:
                mean_pop = 0.5 * (N_lo + N_up)
                density = mean_pop / current_L
                if density > 1e-10:
                    return density, current_L, True
            
            sim_lo.run_events(max(10, (N_lo + N_up) // 4))
            sim_up.run_events(max(10, (N_lo + N_up) // 4))
        else:
            # Didn't converge but got some estimate
            N_lo, N_up = sim_lo.current_population(), sim_up.current_population()
            if N_lo > 0 and N_up > 0:
                density = 0.5 * (N_lo + N_up) / current_L
                return density, current_L, False
    
    return 0.0, current_L, False


def run_measurement_phase(
    d: float, d_prime: float,
    calibration_density: float,
    target_pop: int = 1000,
    max_char_times: int = 1000,
    target_rel_error: float = 0.01,
    seed: int = 42,
    verbose: bool = False,
    diagnostic_writer: Optional[DiagnosticWriter] = None
) -> dict:
    """
    Phase 2: Run final measurement with area scaled for target population.
    
    Returns full simulation results dict.
    """
    if calibration_density <= 1e-10 or d >= B:
        return {
            'converged': False, 'L': np.nan, 't_eq': None, 't_cross': None,
            'density_eq': 0.0, 'density_se': np.nan, 'n_eff': 0, 'n_samples': 0,
            'xi': np.nan, 'z_mean': np.nan, 'pcf_pval': np.nan
        }
    
    # Scale area for target population
    L = target_pop / calibration_density
    N_lo_start = max(10, int(0.1 * target_pop))
    N_up_start = int(1.2 * target_pop)
    
    return detect_equilibrium_dual_sim(
        d, d_prime, L, N_lo_start, N_up_start,
        max_char_times=max_char_times,
        target_rel_error=target_rel_error,
        seed=seed,
        verbose=verbose,
        target_eq_population=target_pop,
        diagnostic_writer=diagnostic_writer
    )


def run_single_simulation(
    params: SimulationParams,
    sim_id: int,
    target_pop: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    save_diagnostics: bool = True
) -> SimulationResult:
    """
    Run complete two-phase simulation for a single (d, d') point.
    
    Phase 1 (Calibration): Quick run to estimate equilibrium density
    Phase 2 (Measurement): Full run with area scaled for target_pop ≈ 1000
    
    Args:
        params: Simulation parameters (d, d', is_fine_grid)
        sim_id: Unique simulation identifier
        target_pop: Target equilibrium population
        seed: Random seed for reproducibility
        verbose: Print progress info
        save_diagnostics: Whether to save diagnostic data (trace, PCF, density) to CSV
    
    Returns SimulationResult dataclass.
    """
    d, d_prime = params.d, params.d_prime
    
    # Initial area estimate from mean field
    n_mf = mean_field_density(d, d_prime)
    initial_L = target_pop / n_mf if n_mf > 0 else target_pop * 100
    
    # Create diagnostic writer if enabled
    diag_writer = None
    if save_diagnostics:
        # Create diagnostics directory for this d' value
        d_prime_str = str(d_prime).replace('.', '_')
        diag_dir = Path(__file__).parent / "results" / "diagnostics" / f"dprime_{d_prime_str}"
        diag_writer = DiagnosticWriter(sim_id, diag_dir)
    
    # Phase 1: Calibration (no diagnostics saved)
    cal_density, cal_L, cal_converged = run_calibration_phase(
        d, d_prime, initial_L, seed=seed, verbose=verbose
    )
    
    # Phase 2: Measurement (with diagnostics)
    if cal_density > 1e-10:
        result = run_measurement_phase(
            d, d_prime, cal_density, target_pop=target_pop, 
            seed=seed + 10000, verbose=verbose,
            diagnostic_writer=diag_writer
        )
    else:
        result = {
            'converged': False, 'L': cal_L, 't_eq': None, 't_cross': None,
            'density_eq': 0.0, 'density_se': np.nan, 'n_eff': 0, 'n_samples': 0,
            'xi': np.nan, 'z_mean': np.nan, 'pcf_pval': np.nan
        }
    
    return SimulationResult(
        sim_id=sim_id,
        seed=seed,
        d=d,
        d_prime=d_prime,
        is_fine_grid=params.is_fine_grid,
        density_eq=result.get('density_eq', 0.0),
        density_se=result.get('density_se', np.nan),
        xi=result.get('xi', np.nan),
        t_eq=result.get('t_eq'),
        L=result.get('L', np.nan),
        converged=result.get('converged', False),
        n_eff=result.get('n_eff', 0),
        n_samples=result.get('n_samples', 0),
        calibration_density=cal_density,
        calibration_L=cal_L,
        z_mean=result.get('z_mean', np.nan),
        pcf_pval=result.get('pcf_pval', np.nan)
    )


# Thread lock for CSV writing
import threading
_csv_write_lock = threading.Lock()


# =============================================================================
# Diagnostic Data Writer - Incremental saving during simulation
# =============================================================================
class DiagnosticWriter:
    """
    Incremental writer for diagnostic data during simulation.
    
    Writes trace, PCF, and density data to separate CSV files for each simulation.
    Thread-safe through separate file handles per sim_id (no locking needed).
    """
    
    def __init__(self, sim_id: int, diagnostics_dir: Path):
        """
        Initialize diagnostic writer for a single simulation.
        
        Args:
            sim_id: Unique simulation identifier
            diagnostics_dir: Directory to save diagnostic files
        """
        self.sim_id = sim_id
        self.diagnostics_dir = Path(diagnostics_dir)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_file = self.diagnostics_dir / f"trace_{sim_id}.csv"
        self.pcf_file = self.diagnostics_dir / f"pcf_{sim_id}.csv"
        self.density_file = self.diagnostics_dir / f"density_{sim_id}.csv"
        
        self._init_trace_file()
        self._init_pcf_file()
        self._init_density_file()
        self._init_spatial_file()
    
    def _init_trace_file(self):
        """Initialize trace CSV with header."""
        with open(self.trace_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sim_id', 't', 'pop_lo', 'pop_up', 'stage'])
    
    def _init_pcf_file(self):
        """Initialize PCF CSV with header."""
        with open(self.pcf_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sim_id', 'sample_idx', 'r_bin', 'g_mean', 'g_se', 'n_particles'])
    
    def _init_density_file(self):
        """Initialize density CSV with header."""
        with open(self.density_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sim_id', 'sample_idx', 'density', 'n_lo', 'n_up'])
    
    def _init_spatial_file(self):
        """Initialize spatial distribution CSV with header."""
        self.spatial_file = self.diagnostics_dir / f"spatial_{self.sim_id}.csv"
        with open(self.spatial_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sim_id', 'position', 'L'])
    
    def write_trace_row(self, t: int, pop_lo: int, pop_up: int, stage: str):
        """Append one trace row."""
        with open(self.trace_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.sim_id, t, pop_lo, pop_up, stage])
    
    def write_pcf_sample(self, sample_idx: int, r: NDArray, g_mean: NDArray, 
                         g_se: NDArray, n_particles: int):
        """Write PCF data for one sample (multiple r bins)."""
        with open(self.pcf_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for r_val, g_val, se_val in zip(r, g_mean, g_se):
                writer.writerow([self.sim_id, sample_idx, r_val, g_val, se_val, n_particles])
    
    def write_density_sample(self, sample_idx: int, density: float, n_lo: int, n_up: int):
        """Append one density sample row."""
        with open(self.density_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.sim_id, sample_idx, density, n_lo, n_up])
    
    def write_spatial_distribution(self, positions: NDArray, L: float):
        """Write spatial positions at start of sampling phase."""
        with open(self.spatial_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for pos in positions.flatten():
                writer.writerow([self.sim_id, pos, L])


class PCFRunningStats:
    """
    Online statistics for PCF averaging using Welford's algorithm.
    
    Computes mean and standard error without storing all samples.
    """
    
    def __init__(self, n_bins: int):
        """
        Initialize running statistics.
        
        Args:
            n_bins: Number of radial bins for PCF
        """
        self.n = 0
        self.mean = np.zeros(n_bins)
        self.M2 = np.zeros(n_bins)  # Running sum of squared differences
        self.r = None  # r-bin centers (set on first sample)
    
    def update(self, r: NDArray, g: NDArray):
        """
        Update statistics with new PCF sample.
        
        Args:
            r: Radial bin centers
            g: PCF values g(r)
        """
        if self.r is None:
            self.r = r.copy()
        
        self.n += 1
        delta = g - self.mean
        self.mean += delta / self.n
        delta2 = g - self.mean
        self.M2 += delta * delta2
    
    def get_mean_and_se(self) -> Tuple[NDArray, NDArray]:
        """
        Return mean and standard error.
        
        Returns:
            (mean_g, se_g) where both are arrays of shape (n_bins,)
        """
        if self.n < 2:
            return self.mean, np.zeros_like(self.mean)
        variance = self.M2 / (self.n - 1)
        se = np.sqrt(variance / self.n)
        return self.mean, se


def _write_result_to_csv(result: SimulationResult, csv_path: Path):
    """Thread-safe: append single result to CSV file."""
    with _csv_write_lock:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSVResultsWriter.FIELDNAMES)
            row = asdict(result)
            row = {k: ('' if v is None else v) for k, v in row.items()}
            writer.writerow(row)


def run_all_simulations_parallel(
    params_list: List[SimulationParams],
    csv_path: Path,
    target_pop: int = 1000,
    n_jobs: int = 1,
    verbose: bool = False
) -> List[SimulationResult]:
    """
    Run all simulations in parallel and save results incrementally to CSV.
    
    Results are written to CSV immediately as each simulation completes,
    ensuring durability even if the script crashes.
    
    Args:
        params_list: List of SimulationParams to simulate
        csv_path: Path to output CSV file
        target_pop: Target equilibrium population
        n_jobs: Number of parallel workers
        verbose: Print progress info
    
    Returns:
        List of SimulationResult
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete existing CSV and create fresh with header
    if csv_path.exists():
        csv_path.unlink()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSVResultsWriter.FIELDNAMES)
        writer.writeheader()
    
    print(f"\nRunning {len(params_list)} simulations...")
    print(f"  Target population: {target_pop}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Output: {csv_path}")
    
    def run_and_save(idx: int, params: SimulationParams) -> SimulationResult:
        """Run simulation and immediately save to CSV."""
        seed = 42 + idx * 100
        try:
            result = run_single_simulation(params, sim_id=idx, target_pop=target_pop, seed=seed, verbose=verbose)
            _write_result_to_csv(result, csv_path)
            return result
        except Exception as e:
            print(f"Error in simulation d={params.d}, d'={params.d_prime}: {e}")
            # Return a failed result
            result = SimulationResult(
                sim_id=idx, seed=seed,
                d=params.d, d_prime=params.d_prime, is_fine_grid=params.is_fine_grid,
                density_eq=0.0, density_se=np.nan, xi=np.nan, t_eq=None, L=np.nan,
                converged=False, n_eff=0, n_samples=0, calibration_density=0.0,
                calibration_L=0.0, z_mean=np.nan, pcf_pval=np.nan
            )
            _write_result_to_csv(result, csv_path)
            return result
    
    if n_jobs == 1:
        results = []
        for idx, params in enumerate(tqdm(params_list, desc="Simulating")):
            result = run_and_save(idx, params)
            results.append(result)
    else:
        # Run all in parallel - each job saves immediately to CSV
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_and_save)(idx, params)
            for idx, params in enumerate(tqdm(params_list, desc="Simulating"))
        )
    
    print(f"  Saved {len(results)} results to {csv_path}")
    return results


# =============================================================================
# Plotting - Summary and Analysis
# =============================================================================
def plot_summary(
    results: List[SimulationResult],
    output_dir: Path,
    show: bool = False
):
    """
    Generate summary plot for extinction scaling results.
    
    Summary Plot Layout (2x2):
    ==========================
    
    1. Steps to Equilibrium (top-left):
       Bar chart showing characteristic times to reach equilibrium vs death rate d,
       grouped by d' value. Reveals how convergence slows near extinction threshold.
       Useful for understanding computational cost and system dynamics.
       
    2. Power-Law Fit (top-right):
       Density vs d with fitted power-law curves n = A(d_ext - d)^β for each d'.
       Shows fitted extinction threshold d_ext with vertical dashed lines.
       Legend displays fit parameters (d_ext, β) and R² goodness-of-fit.
       Key plot for identifying critical extinction threshold.
       
    3. Density Scaling (bottom-left):
       Log-log plot of equilibrium density vs (1-d) with 95% CI error bars.
       Compares measured density against mean-field prediction n_mf = (b-d)/d'.
       Deviations from mean-field indicate importance of spatial correlations.
       
    4. Correlation Length (bottom-right):
       Correlation length ξ (from exponential PCF fit) vs d for each d'.
       Shows how spatial clustering evolves as extinction is approached.
       Diverging ξ suggests critical behavior near extinction.
    
    Args:
        results: List of SimulationResult from completed simulations
        output_dir: Directory to save plots
        show: Display plot interactively
    """
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by d'
    dprime_groups = {}
    for r in results:
        if r.d_prime not in dprime_groups:
            dprime_groups[r.d_prime] = []
        dprime_groups[r.d_prime].append(r)
    
    # Sort each group by d
    for dp in dprime_groups:
        dprime_groups[dp].sort(key=lambda x: x.d)
    
    colors = {1.0: 'blue', 0.1: 'green', 0.01: 'red'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Steps to equilibrium
    ax = axes[0, 0]
    width = 0.02
    offsets = {1.0: -width, 0.1: 0, 0.01: width}
    for d_prime, group in sorted(dprime_groups.items()):
        d_vals = np.array([r.d for r in group])
        t_eq_vals = np.array([r.t_eq if r.t_eq is not None else np.nan for r in group])
        valid = np.isfinite(t_eq_vals)
        c = colors.get(d_prime, 'black')
        offset = offsets.get(d_prime, 0)
        ax.bar(d_vals[valid] + offset, t_eq_vals[valid], width=width, 
               color=c, alpha=0.7, label=f"d'={d_prime}")
    ax.set_xlabel('d (death rate)')
    ax.set_ylabel('Steps to equilibrium')
    ax.set_title('Convergence Time vs Death Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Power-Law Fit with extinction threshold (using log(1+density) y-axis)
    ax = axes[0, 1]
    fit_results = {}
    for d_prime, group in sorted(dprime_groups.items()):
        d_vals = np.array([r.d for r in group])
        density = np.array([r.density_eq for r in group])
        density_se = np.array([r.density_se for r in group])
        fine_mask = np.array([r.is_fine_grid for r in group])
        
        c = colors.get(d_prime, 'black')
        
        # Plot ALL data points using log(1+density) transform - includes zero density
        y_all = np.log1p(density)  # log(1+n) handles n=0 gracefully
        # Error bars in transformed space: d(log(1+n))/dn = 1/(1+n), so se_transformed ≈ se/(1+n)
        yerr_all = np.where(np.isfinite(density_se), 1.96 * density_se / (1 + density), 0)
        ax.errorbar(d_vals, y_all, yerr=yerr_all, fmt='o', color=c,
                   capsize=2, alpha=0.5, markersize=4)
        
        # Fit power law
        fit = fit_power_law_extinction(d_vals, density, density_se, fine_mask)
        fit_results[d_prime] = fit
        
        # Plot fitted curve in log(1+n) space
        if np.isfinite(fit.get('d_ext', np.nan)) and np.isfinite(fit.get('beta', np.nan)):
            d_fit_curve = np.linspace(0, fit['d_ext'] - 0.001, 100)
            n_fit_curve = fit['A'] * np.maximum(fit['d_ext'] - d_fit_curve, 1e-10)**fit['beta']
            ax.plot(d_fit_curve, np.log1p(n_fit_curve), '-', color=c, linewidth=2,
                   label=f"d'={d_prime}: d_ext={fit['d_ext']:.3f}, β={fit['beta']:.2f}, R²={fit['r_squared']:.2f}")
            ax.axvline(fit['d_ext'], color=c, linestyle=':', alpha=0.5)
    
    ax.set_xlabel('d (death rate)')
    ax.set_ylabel('log(1 + density)')
    ax.set_title('Power-Law Fit: n = A(d_ext - d)^β [log(1+n) scale]')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    # 3. Density scaling (log-log)
    ax = axes[1, 0]
    for d_prime, group in sorted(dprime_groups.items()):
        d_vals = np.array([r.d for r in group])
        density = np.array([r.density_eq for r in group])
        density_se = np.array([r.density_se for r in group])
        
        c = colors.get(d_prime, 'black')
        valid = density > 0
        x = 1 - d_vals[valid]
        y = density[valid]
        yerr = np.where(np.isfinite(density_se[valid]), 1.96 * density_se[valid], 0)
        
        ax.errorbar(x, y, yerr=yerr, fmt='o', color=c, capsize=3, alpha=0.7,
                   label=f"d'={d_prime}")
        
        # Mean field prediction
        mf = np.array([mean_field_density(d, d_prime) for d in d_vals[valid]])
        ax.plot(x, mf, '--', color=c, alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('1 - d')
    ax.set_ylabel('Density')
    ax.set_title('Density Scaling (log-log), dashed=mean field')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Correlation length
    ax = axes[1, 1]
    for d_prime, group in sorted(dprime_groups.items()):
        d_vals = np.array([r.d for r in group])
        xi_vals = np.array([r.xi for r in group])
        
        c = colors.get(d_prime, 'black')
        valid = np.isfinite(xi_vals) & (xi_vals > 0)
        ax.plot(d_vals[valid], xi_vals[valid], 'o-', color=c, alpha=0.7,
               label=f"d'={d_prime}")
    
    ax.axhline(SIGMA, color='gray', linestyle=':', alpha=0.5, label='σ (kernel width)')
    ax.set_xlabel('d (death rate)')
    ax.set_ylabel('ξ (correlation length)')
    ax.set_title('Correlation Length vs Death Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Extinction Scaling Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    print(f"Saved: {output_dir / 'summary.png'}")
    return fit_results


def plot_density_vs_extinction_distance(
    results: List[SimulationResult],
    fit_results: dict,
    output_dir: Path,
    show: bool = False
):
    """
    Generate log-log plots of density vs distance from extinction threshold.
    
    Creates one plot per d' value showing:
    - x-axis: log(d_ext - d) where d_ext is the fitted extinction threshold
    - y-axis: log(density)
    - Shows only converged simulations with positive density
    - Power-law relationship appears as straight line
    - Includes fitted power-law curve for reference
    
    Args:
        results: List of SimulationResult
        fit_results: Dict of power-law fits per d' value
        output_dir: Directory to save plots
        show: Display plot interactively
    """
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by d'
    dprime_groups = {}
    for r in results:
        if r.d_prime not in dprime_groups:
            dprime_groups[r.d_prime] = []
        dprime_groups[r.d_prime].append(r)
    
    # Sort each group by d
    for dp in dprime_groups:
        dprime_groups[dp].sort(key=lambda x: x.d)
    
    # Create 1x3 grid (one subplot per d' value)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {1.0: 'blue', 0.1: 'green', 0.01: 'red'}
    
    for idx, (d_prime, group) in enumerate(sorted(dprime_groups.items())):
        ax = axes[idx]
        
        # Get fitted extinction threshold
        fit = fit_results.get(d_prime, {})
        d_ext = fit.get('d_ext', np.nan)
        
        if not np.isfinite(d_ext):
            ax.text(0.5, 0.5, f"d'={d_prime}\nNo valid fit", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"d' = {d_prime}")
            continue
        
        # Extract data for ALL converged simulations with positive density (for plotting)
        d_vals_all = np.array([r.d for r in group if r.converged and r.density_eq > 0])
        density_all = np.array([r.density_eq for r in group if r.converged and r.density_eq > 0])
        density_se_all = np.array([r.density_se for r in group if r.converged and r.density_eq > 0])
        is_fine_all = np.array([r.is_fine_grid for r in group if r.converged and r.density_eq > 0])
        
        # Extract FINE GRID data only (for fitting)
        d_vals_fine = np.array([r.d for r in group if r.converged and r.density_eq > 0 and r.is_fine_grid])
        density_fine = np.array([r.density_eq for r in group if r.converged and r.density_eq > 0 and r.is_fine_grid])
        density_se_fine = np.array([r.density_se for r in group if r.converged and r.density_eq > 0 and r.is_fine_grid])
        
        # Plot ALL data points (coarse + fine)
        distance_all = d_ext - d_vals_all
        valid_all = distance_all > 0
        
        if np.sum(valid_all) == 0:
            ax.text(0.5, 0.5, f"d'={d_prime}\nNo data before extinction", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"d' = {d_prime}")
            continue
        
        x_all = distance_all[valid_all]
        y_all = density_all[valid_all]
        y_err_all = density_se_all[valid_all]
        is_fine_plot = is_fine_all[valid_all]
        
        # Plot data points with different markers for coarse vs fine grid
        c = colors.get(d_prime, 'black')
        yerr = np.where(np.isfinite(y_err_all), 1.96 * y_err_all, 0)
        
        # Plot fine grid points (filled circles)
        fine_mask = is_fine_plot
        if np.sum(fine_mask) > 0:
            ax.errorbar(x_all[fine_mask], y_all[fine_mask], yerr=yerr[fine_mask], 
                       fmt='o', color=c, capsize=3, alpha=0.7, markersize=6, 
                       label='Fine grid')
        
        # Plot coarse grid points (open circles)
        coarse_mask = ~is_fine_plot
        if np.sum(coarse_mask) > 0:
            ax.errorbar(x_all[coarse_mask], y_all[coarse_mask], yerr=yerr[coarse_mask], 
                       fmt='o', markerfacecolor='none', markeredgecolor=c, 
                       capsize=3, alpha=0.5, markersize=6, label='Coarse grid')
        
        # Plot fitted power-law curve
        A = fit.get('A', np.nan)
        beta = fit.get('beta', np.nan)
        r_sq = fit.get('r_squared', np.nan)
        
        if np.isfinite(A) and np.isfinite(beta) and len(x_all) > 0:
            x_fit = np.logspace(np.log10(min(x_all)*0.5), np.log10(max(x_all)*1.5), 100)
            y_fit = A * x_fit**beta
            ax.plot(x_fit, y_fit, '-', color=c, linewidth=2, alpha=0.8,
                   label=f'Fit: n = {A:.2f}·Δd^{beta:.2f}\nR² = {r_sq:.3f}')
        
        # Set log-log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel('d_ext - d  (distance from extinction)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f"d' = {d_prime},  d_ext = {d_ext:.3f}", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle('Density vs Distance from Extinction Threshold (Log-Log)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'density_vs_extinction_distance.png', dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    print(f"Saved: {output_dir / 'density_vs_extinction_distance.png'}")

def load_diagnostic_trace(trace_file: Path) -> dict:
    """Load population trace from diagnostic CSV."""
    if not trace_file.exists():
        return None
    
    data = {'t': [], 'pop_lo': [], 'pop_up': [], 'stage': []}
    with open(trace_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['t'].append(int(row['t']))
            data['pop_lo'].append(int(row['pop_lo']))
            data['pop_up'].append(int(row['pop_up']))
            data['stage'].append(row['stage'])
    
    return {k: np.array(v) if k != 'stage' else v for k, v in data.items()}


def load_diagnostic_density(density_file: Path) -> dict:
    """Load density samples from diagnostic CSV."""
    if not density_file.exists():
        return None
    
    data = {'sample_idx': [], 'density': [], 'n_lo': [], 'n_up': []}
    with open(density_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['sample_idx'].append(int(row['sample_idx']))
            data['density'].append(float(row['density']))
            data['n_lo'].append(int(row['n_lo']))
            data['n_up'].append(int(row['n_up']))
    
    return {k: np.array(v) for k, v in data.items()}


def load_diagnostic_pcf(pcf_file: Path) -> dict:
    """Load PCF data from diagnostic CSV."""
    if not pcf_file.exists():
        return None
    
    # PCF file has multiple rows per sample_idx (one per r_bin)
    data = {}
    with open(pcf_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_idx = int(row['sample_idx'])
            if sample_idx not in data:
                data[sample_idx] = {'r': [], 'g_mean': [], 'g_se': [], 'n_particles': int(row['n_particles'])}
            data[sample_idx]['r'].append(float(row['r_bin']))
            data[sample_idx]['g_mean'].append(float(row['g_mean']))
            data[sample_idx]['g_se'].append(float(row['g_se']))
    
    # Use the last (final) sample for plotting
    if not data:
        return None
    
    final_sample = data[max(data.keys())]
    return {
        'r': np.array(final_sample['r']),
        'g_mean': np.array(final_sample['g_mean']),
        'g_se': np.array(final_sample['g_se']),
        'n_particles': final_sample['n_particles']
    }


def plot_population_dynamics_grid(
    results: List[SimulationResult],
    d_prime: float,
    diagnostics_dir: Path,
    output_dir: Path,
    show: bool = False
):
    """
    Generate 2x3 grid plots of population dynamics for all d values.
    
    Each subplot shows:
    - Blue line: low initial population trace
    - Red line: high initial population trace
    - Green vertical line: crossing point
    - Purple vertical line: equilibrium point
    - Gray horizontal line: mean-field prediction
    
    All d values for this d' are plotted in groups of 6.
    """
    import matplotlib.pyplot as plt
    
    # Filter results for this d' and sort by d
    d_results = [r for r in results if r.d_prime == d_prime and r.converged]
    d_results.sort(key=lambda x: x.d)
    
    if not d_results:
        print(f"No converged results for d'={d_prime}")
        return
    
    n_d = len(d_results)
    d_per_plot = 6
    n_plots = (n_d + d_per_plot - 1) // d_per_plot
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for plot_idx in range(n_plots):
        start_i = plot_idx * d_per_plot
        end_i = min(start_i + d_per_plot, n_d)
        n_subplots = end_i - start_i
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for subplot_idx, i in enumerate(range(start_i, end_i)):
            ax = axes[subplot_idx]
            result = d_results[i]
            
            # Load trace data
            trace_file = diagnostics_dir / f"trace_{result.sim_id}.csv"
            trace_data = load_diagnostic_trace(trace_file)
            
            if trace_data is None:
                ax.text(0.5, 0.5, f'd={result.d:.4f}\n(no trace data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'd = {result.d:.4f}')
                continue
            
            # Plot population traces
            ax.plot(trace_data['t'], trace_data['pop_lo'], 'b-', linewidth=1, alpha=0.8, label='Low start')
            ax.plot(trace_data['t'], trace_data['pop_up'], 'r-', linewidth=1, alpha=0.8, label='High start')
            
            # Mark equilibrium point
            if result.t_eq is not None:
                ax.axvline(result.t_eq, color='purple', linestyle=':', alpha=0.7, label=f'Eq @ {result.t_eq}')
            
            # Mean-field expectation
            n_mf = mean_field_density(result.d, d_prime)
            pop_exp = n_mf * result.L
            ax.axhline(pop_exp, color='gray', linestyle=':', alpha=0.5, label=f'MF: {pop_exp:.0f}')
            
            ax.set_xlabel('Time step')
            ax.set_ylabel('Population')
            ax.set_title(f'd = {result.d:.4f}, L = {result.L:.0f}')
            ax.grid(True, alpha=0.3)
            
            if subplot_idx == 0:
                ax.legend(loc='best', fontsize=7)
        
        # Hide unused axes
        for idx in range(n_subplots, 6):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Population Dynamics (d\'={d_prime})', fontsize=14)
        plt.tight_layout()
        
        filename = f'pop_dynamics_grid_{plot_idx+1}.png'
        plt.savefig(output_dir / filename, dpi=150)
        print(f"Saved: {output_dir / filename}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_density_distribution_grid(
    results: List[SimulationResult],
    d_prime: float,
    diagnostics_dir: Path,
    output_dir: Path,
    show: bool = False
):
    """
    Generate 2x3 grid plots of density distributions during sampling phase.
    
    Each subplot shows:
    - Histogram of density samples
    - Vertical line: measured mean density
    - Dashed line: mean-field prediction
    
    All d values for this d' are plotted in groups of 6.
    """
    import matplotlib.pyplot as plt
    
    # Filter results for this d' and sort by d
    d_results = [r for r in results if r.d_prime == d_prime and r.converged]
    d_results.sort(key=lambda x: x.d)
    
    if not d_results:
        print(f"No converged results for d'={d_prime}")
        return
    
    n_d = len(d_results)
    d_per_plot = 6
    n_plots = (n_d + d_per_plot - 1) // d_per_plot
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for plot_idx in range(n_plots):
        start_i = plot_idx * d_per_plot
        end_i = min(start_i + d_per_plot, n_d)
        n_subplots = end_i - start_i
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for subplot_idx, i in enumerate(range(start_i, end_i)):
            ax = axes[subplot_idx]
            result = d_results[i]
            
            # Load density data
            density_file = diagnostics_dir / f"density_{result.sim_id}.csv"
            density_data = load_diagnostic_density(density_file)
            
            if density_data is None:
                ax.text(0.5, 0.5, f'd={result.d:.4f}\n(no density data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'd = {result.d:.4f}')
                continue
            
            # Plot histogram
            densities = density_data['density']
            ax.hist(densities, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Measured mean
            ax.axvline(result.density_eq, color='red', linestyle='-', linewidth=2, 
                      label=f'Mean: {result.density_eq:.4f}')
            
            # Mean-field prediction
            n_mf = mean_field_density(result.d, d_prime)
            ax.axvline(n_mf, color='green', linestyle='--', linewidth=2,
                      label=f'MF: {n_mf:.4f}')
            
            ax.set_xlabel('Density')
            ax.set_ylabel('Probability density')
            ax.set_title(f'd = {result.d:.4f}, n = {len(densities)} samples')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        
        # Hide unused axes
        for idx in range(n_subplots, 6):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Density Distribution at Sampling (d\'={d_prime})', fontsize=14)
        plt.tight_layout()
        
        filename = f'density_hist_grid_{plot_idx+1}.png'
        plt.savefig(output_dir / filename, dpi=150)
        print(f"Saved: {output_dir / filename}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_pcf_grid(
    results: List[SimulationResult],
    d_prime: float,
    diagnostics_dir: Path,
    output_dir: Path,
    show: bool = False
):
    """
    Generate 2x3 grid plots of averaged PCF with 95% confidence bands.
    
    Each subplot shows:
    - Averaged g(r) from all samples
    - Shaded region: 95% CI (mean ± 1.96×SE)
    - Horizontal line at g=1 (ideal gas reference)
    - Title includes correlation length ξ
    
    All d values for this d' are plotted in groups of 6.
    """
    import matplotlib.pyplot as plt
    
    # Filter results for this d' and sort by d
    d_results = [r for r in results if r.d_prime == d_prime and r.converged]
    d_results.sort(key=lambda x: x.d)
    
    if not d_results:
        print(f"No converged results for d'={d_prime}")
        return
    
    n_d = len(d_results)
    d_per_plot = 6
    n_plots = (n_d + d_per_plot - 1) // d_per_plot
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for plot_idx in range(n_plots):
        start_i = plot_idx * d_per_plot
        end_i = min(start_i + d_per_plot, n_d)
        n_subplots = end_i - start_i
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for subplot_idx, i in enumerate(range(start_i, end_i)):
            ax = axes[subplot_idx]
            result = d_results[i]
            
            # Load PCF data
            pcf_file = diagnostics_dir / f"pcf_{result.sim_id}.csv"
            pcf_data = load_diagnostic_pcf(pcf_file)
            
            if pcf_data is None:
                ax.text(0.5, 0.5, f'd={result.d:.4f}\n(no PCF data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'd = {result.d:.4f}')
                continue
            
            r = pcf_data['r']
            g_mean = pcf_data['g_mean']
            g_se = pcf_data['g_se']
            
            # Plot mean PCF
            ax.plot(r, g_mean, 'b-', linewidth=2, label='g(r)')
            
            # 95% confidence interval
            ci_lower = g_mean - 1.96 * g_se
            ci_upper = g_mean + 1.96 * g_se
            ax.fill_between(r, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
            
            # Reference line at g=1
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='g=1 (ideal gas)')
            
            # Correlation length indicator
            xi = result.xi if np.isfinite(result.xi) else SIGMA
            ax.axvline(xi, color='red', linestyle=':', alpha=0.7, label=f'ξ={xi:.2f}')
            
            ax.set_xlabel('r (distance)')
            ax.set_ylabel('g(r)')
            ax.set_title(f'd = {result.d:.4f}, ξ = {xi:.2f}, N = {pcf_data["n_particles"]}')
            ax.set_xlim(0, max(r))
            ax.set_ylim(0, max(2.0, np.max(ci_upper) * 1.1))
            ax.grid(True, alpha=0.3)
            
            if subplot_idx == 0:
                ax.legend(loc='best', fontsize=7)
        
        # Hide unused axes
        for idx in range(n_subplots, 6):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Pair Correlation Function g(r) with 95% CI (d\'={d_prime})', fontsize=14)
        plt.tight_layout()
        
        filename = f'pcf_grid_{plot_idx+1}.png'
        plt.savefig(output_dir / filename, dpi=150)
        print(f"Saved: {output_dir / filename}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


def generate_diagnostic_plots(
    results: List[SimulationResult],
    output_dir: Path,
    show: bool = False
):
    """
    Generate all diagnostic plots for each d' value.
    
    Creates three types of plots in 2x3 grids:
    1. Population dynamics (trace plots)
    2. Density distribution (histograms)
    3. Pair correlation function with confidence bands
    
    All d values are plotted in groups of 6 or fewer.
    """
    output_dir = Path(output_dir)
    
    # Group results by d'
    dprime_values = sorted(set(r.d_prime for r in results))
    
    for d_prime in dprime_values:
        d_prime_str = str(d_prime).replace('.', '_')
        diag_dir = output_dir / "diagnostics" / f"dprime_{d_prime_str}"
        plot_dir = diag_dir  # Save plots in same directory as diagnostic CSVs
        
        if not diag_dir.exists():
            print(f"No diagnostic data found for d'={d_prime} in {diag_dir}")
            continue
        
        print(f"\nGenerating diagnostic plots for d'={d_prime}...")
        
        # Generate three types of plots
        plot_population_dynamics_grid(results, d_prime, diag_dir, plot_dir, show)
        plot_density_distribution_grid(results, d_prime, diag_dir, plot_dir, show)
        plot_pcf_grid(results, d_prime, diag_dir, plot_dir, show)
        
        print(f"All diagnostic plots saved to: {plot_dir}")


def print_results_summary(results: List[SimulationResult]):
    """Print summary statistics for simulation results."""
    # Group by d'
    dprime_groups = {}
    for r in results:
        if r.d_prime not in dprime_groups:
            dprime_groups[r.d_prime] = []
        dprime_groups[r.d_prime].append(r)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for d_prime in sorted(dprime_groups.keys()):
        group = dprime_groups[d_prime]
        n_total = len(group)
        n_converged = sum(1 for r in group if r.converged)
        n_fine = sum(1 for r in group if r.is_fine_grid)
        
        densities = [r.density_eq for r in group if r.density_eq > 0]
        xi_vals = [r.xi for r in group if np.isfinite(r.xi)]
        
        print(f"\nd' = {d_prime}:")
        print(f"  Total points: {n_total} ({n_fine} fine grid)")
        print(f"  Converged: {n_converged}/{n_total}")
        if densities:
            print(f"  Density range: {min(densities):.4f} - {max(densities):.4f}")
        if xi_vals:
            print(f"  ξ range: {min(xi_vals):.2f} - {max(xi_vals):.2f}")


def main():
    """
    Main entry point for extinction scaling experiment.
    
    Usage:
        # Run all d' values in parallel
        python extinction_scaling.py --all --jobs 4
        
        # Run single d' value
        python extinction_scaling.py --dprime 0.01 --jobs 4
        
        # Analyze existing results
        python extinction_scaling.py --analyze results/results.csv
    """
    parser = argparse.ArgumentParser(
        description="Extinction Limit Scaling Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extinction_scaling.py --all --jobs 4
      Run all d' values (1.0, 0.1, 0.01) in parallel with 4 workers

  python extinction_scaling.py --dprime 0.1 --jobs 2
      Run only d'=0.1 with 2 workers

  python extinction_scaling.py --analyze results/results.csv
      Analyze and plot existing results from CSV file
        """
    )
    parser.add_argument('--all', action='store_true',
                       help="Run simulations for all d' values (1.0, 0.1, 0.01)")
    parser.add_argument('--dprime', '-d', type=float, nargs='*',
                       help="Specific d' value(s) to simulate")
    parser.add_argument('--target-pop', '-p', type=int, default=1000,
                       help="Target equilibrium population (default: 1000)")
    parser.add_argument('--jobs', '-j', type=int, default=1,
                       help="Number of parallel jobs (default: 1)")
    parser.add_argument('--output', '-o', type=str, default='results/results.csv',
                       help="Output CSV path (default: results/results.csv)")
    parser.add_argument('--analyze', type=str, default=None,
                       help="Path to existing CSV to analyze (skip simulation)")
    parser.add_argument('--no-plots', action='store_true',
                       help="Skip generating plots")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mode 1: Analyze existing results
    if args.analyze:
        csv_path = Path(args.analyze)
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}")
            return
        print(f"Loading results from {csv_path}...")
        results = load_results_csv(csv_path)
        print(f"Loaded {len(results)} results")
    else:
        # Mode 2: Run simulations
        if args.all:
            dprime_values = DPRIME_VALUES
        elif args.dprime:
            dprime_values = args.dprime
        else:
            dprime_values = [0.01]  # Default
        
        # Create parameter grid
        params = create_parameter_grid(dprime_values=dprime_values)
        print(f"\nParameter grid: {len(params)} points")
        for dp in dprime_values:
            n_dp = sum(1 for p in params if p.d_prime == dp)
            n_fine = sum(1 for p in params if p.d_prime == dp and p.is_fine_grid)
            print(f"  d'={dp}: {n_dp} points ({n_fine} fine grid)")
        
        # Run simulations
        csv_path = output_dir / args.output.replace('results/', '')
        results = run_all_simulations_parallel(
            params, csv_path,
            target_pop=args.target_pop,
            n_jobs=args.jobs
        )
    
    # Print summary
    print_results_summary(results)
    
    # Generate plots
    if not args.no_plots:
        # Generate summary plot
        fit_results = plot_summary(results, output_dir)
        
        # Generate density vs extinction distance plot (1x3 grid)
        plot_density_vs_extinction_distance(results, fit_results, output_dir)
        
        # Generate diagnostic plots (population dynamics, density distribution, PCF)
        print("\n" + "=" * 60)
        print("GENERATING DIAGNOSTIC PLOTS")
        print("=" * 60)
        generate_diagnostic_plots(results, output_dir)
        
        # Print fit results
        print("\n" + "=" * 60)
        print("POWER-LAW FIT RESULTS")
        print("=" * 60)
        for d_prime, fit in sorted(fit_results.items()):
            if np.isfinite(fit.get('d_ext', np.nan)):
                print(f"d' = {d_prime}:")
                print(f"  d_ext = {fit['d_ext']:.4f}")
                print(f"  β = {fit['beta']:.3f}")
                print(f"  R² = {fit['r_squared']:.3f}")


if __name__ == "__main__":
    main()
