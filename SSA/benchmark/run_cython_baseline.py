#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import halfnorm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation import PyGrid1


@dataclass(frozen=True)
class BenchmarkConfig:
    area_len: float = 100.0
    cell_count: int = 200
    birth_rate: float = 1.0
    baseline_death: float = 0.0
    interaction_strength: float = 0.01
    cutoff: float = 5.0
    num_kernel_points: int = 1024
    initial_population: int = 50_000
    warmup_events: int = 10_000
    target_events: int = 10_000_000
    chunk_size: int = 20_000
    rng_seed: int = 2025
    sim_seed: int = 2025
    histogram_bins: int = 100
    realtime_limit_seconds: float = 3600.0


def make_halfnormal_quantile_table(num_points: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    indices = np.arange(num_points, dtype=np.float64)
    quantiles = (indices + 0.5) / num_points
    radii = halfnorm.ppf(quantiles)
    return quantiles, radii


def make_gaussian_kernel_table(support: float, num_points: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    distances = np.linspace(0.0, support, num_points)
    values = 0.5 * halfnorm.pdf(distances)
    values[-1] = 0.0
    return distances, values


def initialize_grid(config: BenchmarkConfig) -> PyGrid1:
    quantiles, radii = make_halfnormal_quantile_table(config.num_kernel_points)
    distances, values = make_gaussian_kernel_table(config.cutoff, config.num_kernel_points)

    grid = PyGrid1(
        M=1,
        areaLen=[config.area_len],
        cellCount=[config.cell_count],
        isPeriodic=False,
        birthRates=[config.birth_rate],
        deathRates=[config.baseline_death],
        ddMatrix=[config.interaction_strength],
        birthX=[quantiles.tolist()],
        birthY=[radii.tolist()],
        deathX_=[[distances.tolist()]],
        deathY_=[[values.tolist()]],
        cutoffs=[config.cutoff],
        seed=config.sim_seed,
        rtimeLimit=config.realtime_limit_seconds,
    )
    return grid


def place_initial_population(grid: PyGrid1, config: BenchmarkConfig) -> None:
    rng = np.random.default_rng(config.rng_seed)
    positions = rng.uniform(0.0, config.area_len, size=config.initial_population)
    coords = [[float(pos)] for pos in positions]
    grid.placePopulation([coords])


def sample_histogram(positions: NDArray[np.float64], bins: int, area_len: float) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    hist, edges = np.histogram(positions, bins=bins, range=(0.0, area_len))
    return hist.astype(np.int64), edges


def render_histogram(
    positions: NDArray[np.float64],
    bins: int,
    area_len: float,
    output_path: Path,
) -> None:
    hist, bin_edges = sample_histogram(positions, bins, area_len)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        centers,
        hist,
        width=(area_len / bins),
        align="center",
        edgecolor="black",
    )
    ax.set_title("Cython baseline particle positions after benchmark")
    ax.set_xlabel("Position")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, area_len)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    benchmark_dir = Path(__file__).resolve().parent
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    grid = initialize_grid(config)
    place_initial_population(grid, config)

    grid.run_events(config.warmup_events)

    start_event_count = grid.event_count

    target_events = int(config.target_events)
    if target_events <= 0:
        first_segment_target = 0
    else:
        first_segment_target = max(1, int(target_events * 0.1))
        if first_segment_target > target_events:
            first_segment_target = target_events
    last_segment_target = target_events - first_segment_target

    first_segment_events = 0
    first_segment_time = 0.0
    last_segment_events = 0
    last_segment_time = 0.0

    performed_events = 0
    start_time = time.perf_counter()

    while performed_events < target_events:
        remaining = target_events - performed_events
        if remaining <= 0:
            break
        if performed_events < first_segment_target:
            segment_remaining = first_segment_target - performed_events
            step = min(config.chunk_size, segment_remaining, remaining)
            current_segment = "first"
        else:
            step = min(config.chunk_size, remaining)
            current_segment = "last"
        if step <= 0:
            break
        step_begin = time.perf_counter()
        before_events = grid.event_count
        grid.run_events(step)
        after_events = grid.event_count
        step_elapsed = time.perf_counter() - step_begin
        performed = after_events - before_events
        if performed <= 0:
            break
        performed_events += performed
        if current_segment == "first":
            first_segment_events += performed
            first_segment_time += step_elapsed
        else:
            last_segment_events += performed
            last_segment_time += step_elapsed

    end_time = time.perf_counter()

    total_events = grid.event_count - start_event_count
    elapsed = end_time - start_time
    throughput = total_events / elapsed if total_events > 0 and elapsed > 0 else 0.0
    first_segment_throughput = (
        first_segment_events / first_segment_time
        if first_segment_events > 0 and first_segment_time > 0
        else 0.0
    )
    last_segment_throughput = (
        last_segment_events / last_segment_time
        if last_segment_events > 0 and last_segment_time > 0
        else 0.0
    )

    coords = grid.get_all_particle_coords()[0]
    positions = np.asarray(coords, dtype=np.float64)
    if positions.ndim > 1:
        positions = positions.reshape(-1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    histogram_path = benchmark_dir / f"cython_histogram_{timestamp}.png"
    render_histogram(positions, config.histogram_bins, config.area_len, histogram_path)

    results = {
        "implementation": "cython",
        "timestamp": timestamp,
        "area_length": config.area_len,
        "cell_count": config.cell_count,
        "initial_population": config.initial_population,
        "warmup_events": config.warmup_events,
        "target_events": config.target_events,
        "performed_events": total_events,
        "elapsed_seconds": elapsed,
        "throughput_events_per_second": throughput,
        "first_segment": {
            "target_events": int(first_segment_target),
            "performed_events": int(first_segment_events),
            "elapsed_seconds": first_segment_time,
            "throughput_events_per_second": first_segment_throughput,
        },
        "last_segment": {
            "target_events": int(last_segment_target),
            "performed_events": int(last_segment_events),
            "elapsed_seconds": last_segment_time,
            "throughput_events_per_second": last_segment_throughput,
        },
        "final_population": grid.total_population,
        "final_birth_rate": grid.total_birth_rate,
        "final_death_rate": grid.total_death_rate,
        "histogram_bins": config.histogram_bins,
        "histogram_path": str(histogram_path),
    }

    results_path = benchmark_dir / f"cython_results_{timestamp}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return results


def main() -> None:
    config = BenchmarkConfig()
    results = run_benchmark(config)
    print(json.dumps(results, indent=2))
    print()
    print(f"Elapsed time: {results['elapsed_seconds']:.3f} s")
    print(f"Throughput: {results['throughput_events_per_second']:.0f} events/s")

    first_segment = results.get("first_segment")
    if first_segment and first_segment.get("performed_events", 0) > 0:
        print(
            f"First 10%: {first_segment['performed_events']:,} events "
            f"in {first_segment['elapsed_seconds']:.3f} s "
            f"({first_segment['throughput_events_per_second']:,.0f} events/s)"
        )

    last_segment = results.get("last_segment")
    if last_segment and last_segment.get("performed_events", 0) > 0:
        print(
            f"Last 90%: {last_segment['performed_events']:,} events "
            f"in {last_segment['elapsed_seconds']:.3f} s "
            f"({last_segment['throughput_events_per_second']:,.0f} events/s)"
        )

    print(f"Final population: {results['final_population']}")
    benchmark_dir = Path(__file__).resolve().parent
    results_path = benchmark_dir / f"cython_results_{results['timestamp']}.json"
    print(f"Results written to: {results_path}")
    print(f"Histogram saved to: {results['histogram_path']}")


if __name__ == "__main__":
    main()