from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import halfnorm

from SSA import (
    make_ssa_state_1d,
    get_all_particle_coords,
)


BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"
BASE_SEED = 2025


def make_halfnormal_quantile_table(
    num_points: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    quantiles = (np.arange(num_points, dtype=np.float64) + 0.5) / num_points
    radii = halfnorm.ppf(quantiles)
    return quantiles, radii


def make_gaussian_kernel_table(
    support: float = 5.0,
    num_points: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.linspace(0.0, support, num_points)
    values = 0.5 * halfnorm.pdf(distances)
    values[-1] = 0.0
    return distances, values


def update_progress(processed: int, total: int, *, width: int = 40) -> None:
    ratio = processed / total if total else 0.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rProgress |{bar}| {ratio * 100:6.2f}%")
    sys.stdout.flush()


def populate_state(sim, positions: np.ndarray) -> None:
    for pos in positions:
        if not sim.spawn_particle(0, float(pos)):
            raise RuntimeError("initial population exceeds configured capacity")


def run_benchmark() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

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
    target_events = 10_000_000
    chunk_size = 20_000

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

    rng = np.random.default_rng(BASE_SEED)
    init_positions = rng.uniform(0.0, area_len, size=initial_population)

    print("SSA 1D benchmark")
    print("----------------")
    print(f"Area length: {area_len}")
    print(f"Initial population: {initial_population}")
    print(f"Warmup events: {warmup_events}")
    print(f"Target events: {target_events}")
    print(f"Cell capacity per cell: {per_cell_capacity}")
    print()

    sim = make_ssa_state_1d(
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

    populate_state(sim, init_positions)

    sim.run_events(warmup_events)

    update_progress(0, target_events)
    performed_events = 0
    start_count = sim.current_event_count()

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

    begin = time.perf_counter()

    while performed_events < target_events:
        remaining = target_events - performed_events
        if remaining <= 0:
            break
        if performed_events < first_segment_target:
            segment_remaining = first_segment_target - performed_events
            step = min(chunk_size, segment_remaining, remaining)
            current_segment = "first"
        else:
            step = min(chunk_size, remaining)
            current_segment = "last"
        if step <= 0:
            break
        step_begin = time.perf_counter()
        before_events = sim.current_event_count()
        sim.run_events(step)
        after_events = sim.current_event_count()
        step_elapsed = time.perf_counter() - step_begin
        performed_step = after_events - before_events
        if performed_step <= 0:
            continue
        performed_events += performed_step
        update_progress(performed_events, target_events)
        if current_segment == "first":
            first_segment_events += performed_step
            first_segment_time += step_elapsed
        else:
            last_segment_events += performed_step
            last_segment_time += step_elapsed

    end = time.perf_counter()
    sys.stdout.write("\n")

    total_elapsed = end - begin
    total_events = sim.current_event_count() - start_count
    throughput = total_events / total_elapsed if total_events > 0 and total_elapsed > 0 else 0.0
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

    positions = get_all_particle_coords(sim)[0]
    bins = 100
    hist, bin_edges = np.histogram(positions, bins=bins, range=(0.0, area_len))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        (bin_edges[:-1] + bin_edges[1:]) / 2.0,
        hist,
        width=(area_len / bins),
        align="center",
        edgecolor="black",
    )
    ax.set_title("Particle position histogram after benchmark")
    ax.set_xlabel("Position")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, area_len)
    fig.tight_layout()

    histogram_path = BENCHMARK_DIR / f"histogram_{timestamp}.png"
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)

    results = {
        "timestamp": timestamp,
        "seed": BASE_SEED,
        "area_length": area_len,
        "cell_count": cell_count,
        "initial_population": initial_population,
        "warmup_events": warmup_events,
        "target_events": target_events,
        "performed_events": int(total_events),
        "elapsed_seconds": total_elapsed,
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
        "final_population": sim.current_population(),
        "final_birth_rate": sim.total_birth_rate,
        "final_death_rate": sim.total_death_rate,
        "histogram_bins": bins,
        "histogram_path": str(histogram_path),
    }

    results_path = BENCHMARK_DIR / f"results_{timestamp}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Elapsed time: {total_elapsed:.3f} s")
    print(f"Throughput: {throughput:,.0f} events/s")
    if first_segment_events > 0:
        print(
            f"First 10%: {first_segment_events:,} events in {first_segment_time:.3f} s "
            f"({first_segment_throughput:,.0f} events/s)"
        )
    if last_segment_events > 0:
        print(
            f"Last 90%: {last_segment_events:,} events in {last_segment_time:.3f} s "
            f"({last_segment_throughput:,.0f} events/s)"
        )
    print(f"Final population: {sim.current_population()}")
    print(f"Results written to: {results_path}")
    print(f"Histogram saved to: {histogram_path}")


if __name__ == "__main__":
    run_benchmark()
