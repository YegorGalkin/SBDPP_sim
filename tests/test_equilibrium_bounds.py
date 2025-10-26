import math
import numpy as np

from SSA import (
    make_ssa_state_1d,
    make_ssa_state_2d,
    make_ssa_state_3d,
)
from test_equal_kernels_nd import (
    half_normal_equal_tables_1d,
    half_normal_equal_tables_2d,
    half_normal_equal_tables_3d,
    STD_B,
    STD_D,
    STD_DD,
    STD_CUTOFF,
    STD_EQ_POPULATION,
)

REL95_MAX = 0.2
WARMUP_1D = 16_000
WARMUP_2D = 20_000
WARMUP_3D = 24_000
BLOCK_EVENTS = 8_000
BLOCKS_1D = 3
BLOCKS_2D = 3
BLOCKS_3D = 3


def _expected_rel95(b: float, d: float, dd_eff: float, volume: float) -> float:
    if b <= d or volume <= 0.0:
        return float("inf")
    return 1.96 * math.sqrt((b * dd_eff) / (((b - d) ** 2) * volume))


def _mean_population(sim, warmup_events: int, block_events: int, blocks: int) -> float:
    if warmup_events > 0:
        sim.run_events(warmup_events)
    samples = np.empty(blocks, dtype=np.float64)
    for i in range(blocks):
        sim.run_events(block_events)
        samples[i] = sim.current_population()
    return float(np.mean(samples)), samples


def test_1d_equilibrium_bound_5pct():
    b = STD_B
    d = STD_D
    dd = STD_DD
    cutoff = STD_CUTOFF
    area_len = 1_000.0
    cells = 1_500

    rel95 = _expected_rel95(b, d, dd, area_len)
    assert rel95 <= REL95_MAX, (
        f"1D rel95={rel95:.4f} > {REL95_MAX:.2f}; b={b}, d={d}, dd={dd}, L={area_len}"
    )

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_1d(cutoff=cutoff)
    sim = make_ssa_state_1d(
        M=1,
        area_len=area_len,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[cutoff]]),
        seed=13579,
        cell_capacity=2048,
        is_periodic=True,
    )

    exp_density = (b - d) / dd
    exp_population = int(exp_density * area_len)
    assert exp_population == STD_EQ_POPULATION

    rng = np.random.default_rng(13579)
    for x in rng.uniform(0.0, area_len, size=exp_population):
        sim.spawn_particle(0, float(x))

    mean_N, samples = _mean_population(sim, WARMUP_1D, BLOCK_EVENTS, BLOCKS_1D)
    density = mean_N / area_len
    lower = exp_density * (1.0 - REL95_MAX)
    upper = exp_density * (1.0 + REL95_MAX)
    assert lower <= density <= upper, (
        f"1D density out of bounds: density={density:.6f}, expected={exp_density:.6f}, "
        f"bounds=[{lower:.6f}, {upper:.6f}], samples={samples}"
    )


def test_2d_equilibrium_bound_5pct():
    b = STD_B
    d = STD_D
    dd = STD_DD
    cutoff = STD_CUTOFF
    Lx, Ly = 25.0, 40.0
    Nx, Ny = 80, 128

    volume = Lx * Ly
    rel95 = _expected_rel95(b, d, dd, volume)
    assert rel95 <= REL95_MAX, (
        f"2D rel95={rel95:.4f} > {REL95_MAX:.2f}; b={b}, d={d}, dd={dd}, V={volume}"
    )

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_2d(cutoff=cutoff)
    sim = make_ssa_state_2d(
        M=1,
        area_len=np.array([Lx, Ly]),
        cell_count=np.array([Nx, Ny]),
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[cutoff]]),
        seed=97531,
        cell_capacity=256,
        is_periodic=True,
    )

    exp_density = (b - d) / dd
    exp_population = int(exp_density * volume)
    assert exp_population == STD_EQ_POPULATION

    rng = np.random.default_rng(97531)
    xs = rng.uniform(0.0, Lx, size=exp_population)
    ys = rng.uniform(0.0, Ly, size=exp_population)
    for x, y in zip(xs, ys):
        sim.spawn_particle(0, float(x), float(y))

    mean_N, samples = _mean_population(sim, WARMUP_2D, BLOCK_EVENTS, BLOCKS_2D)
    density = mean_N / volume
    lower = exp_density * (1.0 - REL95_MAX)
    upper = exp_density * (1.0 + REL95_MAX)
    assert lower <= density <= upper, (
        f"2D density out of bounds: density={density:.6f}, expected={exp_density:.6f}, "
        f"bounds=[{lower:.6f}, {upper:.6f}], samples={samples}"
    )


def test_3d_equilibrium_bound_5pct():
    b = STD_B
    d = STD_D
    dd = STD_DD
    cutoff = STD_CUTOFF
    Lx = Ly = Lz = 10.0
    Nx = Ny = Nz = 30

    volume = Lx * Ly * Lz
    rel95 = _expected_rel95(b, d, dd, volume)
    assert rel95 <= REL95_MAX, (
        f"3D rel95={rel95:.4f} > {REL95_MAX:.2f}; b={b}, d={d}, dd={dd}, V={volume}"
    )

    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_3d(cutoff=cutoff)
    sim = make_ssa_state_3d(
        M=1,
        area_len=np.array([Lx, Ly, Lz]),
        cell_count=np.array([Nx, Ny, Nz]),
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[cutoff]]),
        seed=86420,
        cell_capacity=192,
        is_periodic=True,
    )

    exp_density = (b - d) / dd
    exp_population = int(exp_density * volume)
    assert exp_population == STD_EQ_POPULATION

    rng = np.random.default_rng(86420)
    xs = rng.uniform(0.0, Lx, size=exp_population)
    ys = rng.uniform(0.0, Ly, size=exp_population)
    zs = rng.uniform(0.0, Lz, size=exp_population)
    for x, y, z in zip(xs, ys, zs):
        sim.spawn_particle(0, float(x), float(y), float(z))

    mean_N, samples = _mean_population(sim, WARMUP_3D, BLOCK_EVENTS, BLOCKS_3D)
    density = mean_N / volume
    lower = exp_density * (1.0 - REL95_MAX)
    upper = exp_density * (1.0 + REL95_MAX)
    assert lower <= density <= upper, (
        f"3D density out of bounds: density={density:.6f}, expected={exp_density:.6f}, "
        f"bounds=[{lower:.6f}, {upper:.6f}], samples={samples}"
    )
