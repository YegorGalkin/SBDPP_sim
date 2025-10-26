"""
Benchmark and Profile Script for numba_1d and numba_sim
Runs 1M events (10x less than benchmark_1m_events.py which runs 10M)
Includes profiling and performance comparison
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from scipy.stats import halfnorm

# Import both implementations
from SSA.numba_1d import (
    make_ssa_state_1d as make_ssa_1d,
    get_all_particle_coords as get_coords_1d,
)
from SSA.numba_sim import (
    make_ssa_state_1d as make_ssa_sim,
    get_all_particle_coords as get_coords_sim,
)


BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"
BASE_SEED = 2025


def make_halfnormal_quantile_table(
    num_points: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create quantile table for half-normal distribution."""
    quantiles = (np.arange(num_points, dtype=np.float64) + 0.5) / num_points
    radii = halfnorm.ppf(quantiles)
    return quantiles, radii


def make_gaussian_kernel_table(
    support: float = 5.0,
    num_points: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create Gaussian kernel table."""
    distances = np.linspace(0.0, support, num_points)
    values = 0.5 * halfnorm.pdf(distances)
    values[-1] = 0.0
    return distances, values


def update_progress(processed: int, total: int, *, width: int = 40) -> None:
    """Display a progress bar."""
    ratio = processed / total if total else 0.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rProgress |{bar}| {ratio * 100:6.2f}%")
    sys.stdout.flush()


def populate_state_1d(sim, positions: np.ndarray) -> None:
    """Populate state for numba_1d implementation."""
    for pos in positions:
        if not sim.spawn_particle(0, float(pos)):
            raise RuntimeError("initial population exceeds configured capacity")


def populate_state_sim(sim, positions: np.ndarray) -> None:
    """Populate state for numba_sim implementation."""
    for pos in positions:
        if not sim.spawn_particle(0, float(pos)):
            raise RuntimeError("initial population exceeds configured capacity")


def run_benchmark_1d(
    area_len: float,
    cell_count: int,
    birth_rate: float,
    baseline_death: float,
    interaction_strength: float,
    cutoff: float,
    initial_population: int,
    per_cell_capacity: int,
    warmup_events: int,
    target_events: int,
    chunk_size: int,
    init_positions: np.ndarray,
    birth_x_table: np.ndarray,
    birth_y_table: np.ndarray,
    death_x_table: np.ndarray,
    death_y_table: np.ndarray,
    enable_profiling: bool = False,
) -> dict:
    """Run benchmark for numba_1d implementation."""
    
    print("\n" + "="*60)
    print("BENCHMARKING numba_1d")
    print("="*60)
    
    sim = make_ssa_1d(
        M=1,
        area_len=area_len,
        cell_count=cell_count,
        birth_rates=[birth_rate],
        death_rates=[baseline_death],
        dd_matrix=[[interaction_strength]],
        birth_x=birth_x_table,
        birth_y=birth_y_table,
        death_x=death_x_table,
        death_y=death_y_table,
        cutoffs=[[cutoff]],
        seed=BASE_SEED,
        cell_capacity=per_cell_capacity,
    )

    populate_state_1d(sim, init_positions)
    
    # Warmup
    print(f"Warming up with {warmup_events} events...")
    sim.run_events(warmup_events)
    print(f"After warmup: population = {sim.current_population()}")

    # Main benchmark
    update_progress(0, target_events)
    performed_events = 0
    start_count = sim.current_event_count()

    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    begin = time.perf_counter()

    while performed_events < target_events:
        remaining = target_events - performed_events
        if remaining <= 0:
            break
        step = min(chunk_size, remaining)
        if step <= 0:
            break
        
        before_events = sim.current_event_count()
        sim.run_events(step)
        after_events = sim.current_event_count()
        
        performed_step = after_events - before_events
        if performed_step <= 0:
            continue
        performed_events += performed_step
        update_progress(performed_events, target_events)

    end = time.perf_counter()
    sys.stdout.write("\n")

    if enable_profiling and profiler:
        profiler.disable()

    total_elapsed = end - begin
    total_events = sim.current_event_count() - start_count
    throughput = total_events / total_elapsed if total_events > 0 and total_elapsed > 0 else 0.0

    print(f"Elapsed time: {total_elapsed:.3f} s")
    print(f"Throughput: {throughput:,.0f} events/s")
    print(f"Final population: {sim.current_population()}")
    print(f"Final birth rate: {sim.total_birth_rate:.2f}")
    print(f"Final death rate: {sim.total_death_rate:.2f}")

    # Profiling results
    profile_stats = None
    if enable_profiling and profiler:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        profile_stats = s.getvalue()

    return {
        "implementation": "numba_1d",
        "performed_events": int(total_events),
        "elapsed_seconds": total_elapsed,
        "throughput_events_per_second": throughput,
        "final_population": sim.current_population(),
        "final_birth_rate": sim.total_birth_rate,
        "final_death_rate": sim.total_death_rate,
        "profile_stats": profile_stats,
    }


def run_benchmark_sim(
    area_len: float,
    cell_count: int,
    birth_rate: float,
    baseline_death: float,
    interaction_strength: float,
    cutoff: float,
    initial_population: int,
    per_cell_capacity: int,
    warmup_events: int,
    target_events: int,
    chunk_size: int,
    init_positions: np.ndarray,
    birth_x_table: np.ndarray,
    birth_y_table: np.ndarray,
    death_x_table: np.ndarray,
    death_y_table: np.ndarray,
    enable_profiling: bool = False,
) -> dict:
    """Run benchmark for numba_sim implementation."""
    
    print("\n" + "="*60)
    print("BENCHMARKING numba_sim")
    print("="*60)
    
    sim = make_ssa_sim(
        M=1,
        area_len=area_len,
        birth_rates=[birth_rate],
        death_rates=[baseline_death],
        dd_matrix=[[interaction_strength]],
        birth_x=birth_x_table,
        birth_y=birth_y_table,
        death_x=death_x_table,
        death_y=death_y_table,
        cutoffs=[[cutoff]],
        seed=BASE_SEED,
        cell_count=cell_count,
        cell_capacity=per_cell_capacity,
        is_periodic=False,
    )

    populate_state_sim(sim, init_positions)
    
    # Warmup
    print(f"Warming up with {warmup_events} events...")
    sim.run_events(warmup_events)
    print(f"After warmup: population = {sim.current_population()}")

    # Main benchmark
    update_progress(0, target_events)
    performed_events = 0
    start_count = sim.current_event_count()

    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    begin = time.perf_counter()

    while performed_events < target_events:
        remaining = target_events - performed_events
        if remaining <= 0:
            break
        step = min(chunk_size, remaining)
        if step <= 0:
            break
        
        before_events = sim.current_event_count()
        sim.run_events(step)
        after_events = sim.current_event_count()
        
        performed_step = after_events - before_events
        if performed_step <= 0:
            continue
        performed_events += performed_step
        update_progress(performed_events, target_events)

    end = time.perf_counter()
    sys.stdout.write("\n")

    if enable_profiling and profiler:
        profiler.disable()

    total_elapsed = end - begin
    total_events = sim.current_event_count() - start_count
    throughput = total_events / total_elapsed if total_events > 0 and total_elapsed > 0 else 0.0

    print(f"Elapsed time: {total_elapsed:.3f} s")
    print(f"Throughput: {throughput:,.0f} events/s")
    print(f"Final population: {sim.current_population()}")
    print(f"Final birth rate: {sim.total_birth_rate:.2f}")
    print(f"Final death rate: {sim.total_death_rate:.2f}")

    # Profiling results
    profile_stats = None
    if enable_profiling and profiler:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        profile_stats = s.getvalue()

    return {
        "implementation": "numba_sim",
        "performed_events": int(total_events),
        "elapsed_seconds": total_elapsed,
        "throughput_events_per_second": throughput,
        "final_population": sim.current_population(),
        "final_birth_rate": sim.total_birth_rate,
        "final_death_rate": sim.total_death_rate,
        "profile_stats": profile_stats,
    }


def main():
    """Main benchmark and profiling function."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Configuration (1M events instead of 10M)
    area_len = 100.0
    cell_count = 200
    birth_rate = 1.0
    baseline_death = 0.0
    interaction_strength = 0.01
    cutoff = 5.0

    initial_population = 50_000
    capacity_multiplier = 4
    per_cell_capacity = max(
        1024,
        int(np.ceil(initial_population * capacity_multiplier / cell_count)),
    )

    warmup_events = 10_000
    target_events = 1_000_000  # 1M events (10x less than original)
    chunk_size = 20_000

    # Create kernel tables
    quantiles, radii = make_halfnormal_quantile_table()
    quantiles[0] = 0.0
    radii[0] = 0.0

    death_x_vals, death_y_vals = make_gaussian_kernel_table(cutoff)
    death_x_vals[0] = 0.0
    death_y_vals[0] = 0.0

    birth_x_table = np.ascontiguousarray(quantiles[np.newaxis, :], dtype=np.float64)
    birth_y_table = np.ascontiguousarray(radii[np.newaxis, :], dtype=np.float64)
    death_x_table = np.ascontiguousarray(death_x_vals[np.newaxis, np.newaxis, :], dtype=np.float64)
    death_y_table = np.ascontiguousarray(death_y_vals[np.newaxis, np.newaxis, :], dtype=np.float64)

    # Create initial positions
    rng = np.random.default_rng(BASE_SEED)
    init_positions = rng.uniform(0.0, area_len, size=initial_population)

    print("="*60)
    print("SSA 1D Benchmark and Profiling Comparison")
    print("="*60)
    print(f"Area length: {area_len}")
    print(f"Cell count: {cell_count}")
    print(f"Initial population: {initial_population}")
    print(f"Warmup events: {warmup_events}")
    print(f"Target events: {target_events:,}")
    print(f"Cell capacity per cell: {per_cell_capacity}")
    print()

    # Run benchmarks with profiling
    results_1d = run_benchmark_1d(
        area_len, cell_count, birth_rate, baseline_death, interaction_strength,
        cutoff, initial_population, per_cell_capacity, warmup_events,
        target_events, chunk_size, init_positions.copy(),
        birth_x_table, birth_y_table, death_x_table, death_y_table,
        enable_profiling=True,
    )

    results_sim = run_benchmark_sim(
        area_len, cell_count, birth_rate, baseline_death, interaction_strength,
        cutoff, initial_population, per_cell_capacity, warmup_events,
        target_events, chunk_size, init_positions.copy(),
        birth_x_table, birth_y_table, death_x_table, death_y_table,
        enable_profiling=True,
    )

    # Comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\nnumba_1d:")
    print(f"  Throughput: {results_1d['throughput_events_per_second']:,.0f} events/s")
    print(f"  Time: {results_1d['elapsed_seconds']:.3f} s")
    
    print(f"\nnumba_sim:")
    print(f"  Throughput: {results_sim['throughput_events_per_second']:,.0f} events/s")
    print(f"  Time: {results_sim['elapsed_seconds']:.3f} s")
    
    speedup = results_1d['throughput_events_per_second'] / results_sim['throughput_events_per_second']
    print(f"\nSpeedup (numba_1d / numba_sim): {speedup:.2f}x")
    
    if speedup > 1:
        print(f"numba_1d is {speedup:.2f}x FASTER than numba_sim")
    else:
        print(f"numba_sim is {1/speedup:.2f}x FASTER than numba_1d")

    # Save results
    combined_results = {
        "timestamp": timestamp,
        "configuration": {
            "area_length": area_len,
            "cell_count": cell_count,
            "initial_population": initial_population,
            "warmup_events": warmup_events,
            "target_events": target_events,
            "birth_rate": birth_rate,
            "baseline_death": baseline_death,
            "interaction_strength": interaction_strength,
            "cutoff": cutoff,
        },
        "numba_1d": {
            "throughput": results_1d['throughput_events_per_second'],
            "elapsed": results_1d['elapsed_seconds'],
            "events": results_1d['performed_events'],
            "final_population": results_1d['final_population'],
        },
        "numba_sim": {
            "throughput": results_sim['throughput_events_per_second'],
            "elapsed": results_sim['elapsed_seconds'],
            "events": results_sim['performed_events'],
            "final_population": results_sim['final_population'],
        },
        "speedup": speedup,
    }

    results_path = BENCHMARK_DIR / f"comparison_{timestamp}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(combined_results, fh, indent=2)

    # Save profiling results
    if results_1d['profile_stats']:
        profile_1d_path = BENCHMARK_DIR / f"profile_numba_1d_{timestamp}.txt"
        with profile_1d_path.open("w", encoding="utf-8") as fh:
            fh.write("PROFILING RESULTS FOR numba_1d\n")
            fh.write("="*60 + "\n\n")
            fh.write(results_1d['profile_stats'])
        print(f"\nnumba_1d profiling saved to: {profile_1d_path}")

    if results_sim['profile_stats']:
        profile_sim_path = BENCHMARK_DIR / f"profile_numba_sim_{timestamp}.txt"
        with profile_sim_path.open("w", encoding="utf-8") as fh:
            fh.write("PROFILING RESULTS FOR numba_sim\n")
            fh.write("="*60 + "\n\n")
            fh.write(results_sim['profile_stats'])
        print(f"numba_sim profiling saved to: {profile_sim_path}")

    print(f"\nComparison results saved to: {results_path}")


if __name__ == "__main__":
    main()
