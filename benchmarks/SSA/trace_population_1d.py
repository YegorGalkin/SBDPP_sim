from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SSA import make_ssa_state_1d


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=200.0, help="domain length (1D)")
    p.add_argument("--cells", type=int, default=200, help="number of cells")
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--d", type=float, default=0.0)
    p.add_argument("--dd", type=float, default=0.1)
    p.add_argument("--cutoff", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=13579)
    p.add_argument("--periodic", action="store_true", help="use periodic boundaries")
    p.add_argument("--warmup", type=int, default=20000)
    p.add_argument("--chunks", type=int, default=20)
    p.add_argument("--step", type=int, default=5000)
    args = p.parse_args()

    b = args.b
    d = args.d
    dd = args.dd
    cutoff = args.cutoff
    area_len = args.L
    cell_count = args.cells
    seed = args.seed

    # Build truncated-half-normal equal kernels: death kernel w(r) and birth distance PDF p_r(r)=2*w(r)
    # Choose sigma=1.0 and cutoff in multiples of sigma.
    sigma = 1.0
    rmax = cutoff
    grid = 2048
    r = np.linspace(0.0, rmax, grid, dtype=np.float64)
    # Half-normal pdf (r>=0) with sigma
    p_half = (np.sqrt(2.0) / (sigma * np.sqrt(np.pi))) * np.exp(-(r * r) / (2.0 * sigma * sigma))
    # Base death kernel (before normalization to 1 over R): w_base = p_half/2 (so that 2∫ w = ∫ p_half = 1 on [0,∞))
    w_base = 0.5 * p_half
    # Normalize exactly on [0, cutoff] so that 2∫_0^cutoff w(r)dr = 1
    mass = 2.0 * np.trapz(w_base, r)
    scale = 1.0 / mass if mass > 0 else 1.0
    w = scale * w_base
    p_r = 2.0 * w
    # Inverse CDF table for p_r by numeric inversion
    cdf = np.cumsum((p_r[:-1] + p_r[1:]) * 0.5 * (r[1:] - r[:-1]))
    cdf = np.concatenate(([0.0], cdf))
    # Ensure normalization edge conditions
    if cdf[-1] <= 0.0:
        cdf[-1] = 1.0
    cdf /= cdf[-1]
    num_points = 1024
    quantiles = (np.arange(num_points, dtype=np.float64) + 0.5) / num_points
    # Map quantiles to radii via inverse CDF
    radii = np.interp(quantiles, cdf, r)

    # Tables
    birth_x = quantiles[np.newaxis, :]
    birth_y = radii[np.newaxis, :]
    death_x = r[np.newaxis, np.newaxis, :]
    death_y = w[np.newaxis, np.newaxis, :]

    # With equal normalized kernels and d=0, expected density is b/dd
    exp_density = (b - d) / dd if dd > 0 else 0.0
    init_n = int(exp_density * area_len)

    print(f"Trace 1D | b={b}, d={d}, dd={dd}, cutoff={cutoff}, L={area_len}, N0~{init_n}")

    sim = make_ssa_state_1d(
        M=1,
        area_len=area_len,
        cell_count=cell_count,
        birth_rates=np.array([b], dtype=np.float64),
        death_rates=np.array([d], dtype=np.float64),
        dd_matrix=np.array([[dd]], dtype=np.float64),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[cutoff]], dtype=np.float64),
        seed=seed,
        cell_capacity=1024,
        is_periodic=args.periodic,
    )

    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, area_len, size=init_n)
    for x in xs:
        sim.spawn_particle(0, float(x))

    # Warm-up
    sim.run_events(args.warmup)
    print(f"after warmup={args.warmup}: t={sim.current_time():.6f}, N={sim.current_population()}")

    # Chunked trace
    chunks = args.chunks
    step = args.step
    last_count = sim.current_event_count()
    for i in range(chunks):
        sim.run_events(step)
        now_count = sim.current_event_count()
        performed = now_count - last_count
        last_count = now_count
        print(
            f"chunk {i+1:02d}/{chunks}: +{performed} ev, t={sim.current_time():.6f}, N={sim.current_population()}"
        )

    print(
        f"final: events={sim.current_event_count()}, t={sim.current_time():.6f}, N={sim.current_population()}"
    )


if __name__ == "__main__":
    main()
