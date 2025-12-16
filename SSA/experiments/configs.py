"""
Experiment Configurations for Kappa Estimation.

Three configurations are provided:
- BASELINE: Standard dispersal (sigma=1.0)
- INCREASED_DISPERSAL: Larger dispersal (sigma=2.0, PCF grid 2x larger)
- REDUCED_DISPERSAL: Smaller dispersal (sigma=0.5, PCF grid 2x smaller)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Config name for identification
    name: str = "baseline"
    
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
    
    def get_output_dir(self) -> str:
        """Get output directory with config name."""
        return f"{self.output_dir}/{self.name}"


# =============================================================================
# Predefined Configurations
# =============================================================================

def baseline_config() -> ExperimentConfig:
    """
    Baseline configuration with standard dispersal.
    sigma = 1.0, dr = 0.1, r_max = 5.0
    """
    return ExperimentConfig(
        name="baseline",
        b=1.0,
        d_prime=1.0,
        sigma=1.0,
        d_controls=(0.0, 1e-4, 1e-3),
        d_test_range=(0.01, 0.1, 10),
        dr=0.1,
        r_max=5.0,
    )


def increased_dispersal_config() -> ExperimentConfig:
    """
    Increased dispersal configuration.
    sigma = 2.0 (2x larger)
    PCF grid 2x larger: dr = 0.2, r_max = 10.0
    """
    return ExperimentConfig(
        name="increased_dispersal",
        b=1.0,
        d_prime=1.0,
        sigma=2.0,
        d_controls=(0.0, 1e-4, 1e-3),
        d_test_range=(0.01, 0.1, 10),
        dr=0.2,   # 2x larger
        r_max=10.0,  # 2x larger
    )


def reduced_dispersal_config() -> ExperimentConfig:
    """
    Reduced dispersal configuration.
    sigma = 0.5 (2x smaller)
    PCF grid 2x smaller: dr = 0.05, r_max = 2.5
    """
    return ExperimentConfig(
        name="reduced_dispersal",
        b=1.0,
        d_prime=1.0,
        sigma=0.5,
        d_controls=(0.0, 1e-4, 1e-3),
        d_test_range=(0.01, 0.1, 10),
        dr=0.05,  # 2x smaller
        r_max=2.5,  # 2x smaller
    )


def small_dprime_01_config() -> ExperimentConfig:
    """
    Reduced d' configuration: d'=0.1, L=100.
    All d values reduced 10x proportionally.
    Expected pop at d=0: b/d' * L = 1.0/0.1 * 100 = 1000
    """
    return ExperimentConfig(
        name="small_dprime_01",
        b=1.0,
        d_prime=0.1,  # 10x smaller
        sigma=1.0,
        L=100.0,  # Smaller domain
        d_controls=(0.0, 1e-5, 1e-4),  # 10x smaller
        d_test_range=(0.001, 0.01, 10),  # 10x smaller
        dr=0.1,
        r_max=5.0,
    )


def small_dprime_001_config() -> ExperimentConfig:
    """
    Very reduced d' configuration: d'=0.01, L=50.
    All d values reduced 100x proportionally.
    Expected pop at d=0: b/d' * L = 1.0/0.01 * 50 = 5000
    """
    return ExperimentConfig(
        name="small_dprime_001",
        b=1.0,
        d_prime=0.01,  # 100x smaller
        sigma=1.0,
        L=50.0,  # Smaller domain
        d_controls=(0.0, 1e-6, 1e-5),  # 100x smaller
        d_test_range=(0.0001, 0.001, 10),  # 100x smaller
        dr=0.1,
        r_max=5.0,
    )


# Dictionary of all available configs
CONFIGS = {
    "baseline": baseline_config,
    "increased_dispersal": increased_dispersal_config,
    "reduced_dispersal": reduced_dispersal_config,
    "small_dprime_01": small_dprime_01_config,
    "small_dprime_001": small_dprime_001_config,
}


def get_config(name: str) -> ExperimentConfig:
    """Get configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()


def list_configs() -> list[str]:
    """List available configuration names."""
    return list(CONFIGS.keys())
