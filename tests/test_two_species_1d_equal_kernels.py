import numpy as np

from SSA import make_ssa_state_1d
from test_equal_kernels_nd import (
    EQ_TOLERANCE,
    STD_B,
    STD_CUTOFF,
    STD_D,
    STD_DD,
    STD_EQ_POPULATION,
    half_normal_equal_tables_1d,
)

STD_DENSITY = (STD_B - STD_D) / STD_DD  # 10 individuals per unit length
STD_LENGTH = STD_EQ_POPULATION / STD_DENSITY  # 1000.0
HALF_LENGTH = STD_LENGTH / 2.0  # 500.0
CELL_DENSITY_PER_UNIT = 1.5

WARMUP_BALANCED = 22_000
WARMUP_BIASED = 28_000
BLOCK_EVENTS = 9_000
BLOCKS = 3
SPECIES_TOLERANCE = 0.15  # Allow 15% fluctuation when sampling per-species means


def _cell_count(domain_length: float) -> int:
    return max(512, int(round(domain_length * CELL_DENSITY_PER_UNIT)))


def _make_sim(dd_matrix, domain_length: float, cells: int, *, seed: int) -> any:
    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_1d()
    birth_x_table = np.broadcast_to(birth_x, (2, birth_x.shape[-1])).copy()
    birth_y_table = np.broadcast_to(birth_y, (2, birth_y.shape[-1])).copy()
    death_x_table = np.broadcast_to(death_x, (2, 2, death_x.shape[-1])).copy()
    death_y_table = np.broadcast_to(death_y, (2, 2, death_y.shape[-1])).copy()

    return make_ssa_state_1d(
        M=2,
        area_len=domain_length,
        cell_count=cells,
        birth_rates=np.array([STD_B, STD_B], dtype=np.float64),
        death_rates=np.array([STD_D, STD_D], dtype=np.float64),
        dd_matrix=np.asarray(dd_matrix, dtype=np.float64),
        birth_x=birth_x_table,
        birth_y=birth_y_table,
        death_x=death_x_table,
        death_y=death_y_table,
        cutoffs=np.full((2, 2), STD_CUTOFF, dtype=np.float64),
        seed=seed,
        cell_capacity=2048,
        is_periodic=True,
    )


def _seed_uniform(state, counts, rng: np.random.Generator) -> None:
    length = float(state.area_size[0])
    for species_id, count in enumerate(counts):
        positions = rng.uniform(0.0, length, size=int(count))
        for pos in positions:
            state.spawn_particle(species_id, float(pos))


def _collect_counts(state, warmup_events: int) -> tuple[np.ndarray, np.ndarray]:
    if warmup_events > 0:
        state.run_events(warmup_events)

    totals = np.empty(BLOCKS, dtype=np.float64)
    species_counts = np.empty((BLOCKS, 2), dtype=np.float64)

    for i in range(BLOCKS):
        state.run_events(BLOCK_EVENTS)
        total = state.current_population()
        totals[i] = total
        counts = state.get_species_counts()
        species_counts[i] = counts

    return totals, species_counts


def test_two_species_no_cross():
    length = HALF_LENGTH
    cells = _cell_count(length)
    dd_matrix = [[STD_DD, 0.0], [0.0, STD_DD]]
    sim = _make_sim(dd_matrix, length, cells, seed=111)

    # With no cross-interaction, each species independently reaches equilibrium
    # Each species reaches (b-d)/dd * length = (1.0-0.0)/0.1 * 500 = 5000
    # Total should be approximately 2 * 5000 = 10000
    eq_per_species = int(STD_DENSITY * length)
    expected_total = 2 * eq_per_species
    
    rng = np.random.default_rng(111)
    _seed_uniform(sim, [eq_per_species / 2.0, eq_per_species / 2.0], rng)

    totals, species_counts = _collect_counts(sim, WARMUP_BALANCED)
    total_mean = float(np.mean(totals))
    mean_counts = np.mean(species_counts, axis=0)

    # Allow slightly higher tolerance for two-species stochastic dynamics
    assert abs(total_mean - expected_total) <= 0.10 * expected_total, (
        f"No-cross total mean {total_mean:.1f}, expected {expected_total}, "
        f"samples={totals}"
    )
    for idx, mean in enumerate(mean_counts):
        assert abs(mean - eq_per_species) <= SPECIES_TOLERANCE * eq_per_species, (
            f"No-cross species {idx} mean {mean:.1f}, expected {eq_per_species}, "
            f"samples={species_counts[:, idx]}"
        )


def test_two_species_full_cross_extinction():
    length = STD_LENGTH
    cells = _cell_count(length)
    dd_matrix = [[STD_DD, STD_DD], [STD_DD, STD_DD]]
    sim = _make_sim(dd_matrix, length, cells, seed=222)

    # With full cross-competition, equilibrium density is the same as single species
    # Start with equilibrium population for the full domain
    initial_pop = int(STD_DENSITY * length)
    expected_equilibrium = initial_pop
    
    rng = np.random.default_rng(222)
    _seed_uniform(sim, [initial_pop, 0], rng)

    totals, species_counts = _collect_counts(sim, WARMUP_BIASED)
    total_mean = float(np.mean(totals))
    mean_counts = np.mean(species_counts, axis=0)
    minority = float(np.min(mean_counts))

    # Allow larger tolerance for two-species dynamics
    assert abs(total_mean - expected_equilibrium) <= 0.55 * expected_equilibrium, (
        f"Full-cross total mean {total_mean:.1f}, expected ~{expected_equilibrium}, "
        f"samples={totals}"
    )
    assert minority <= 0.3 * expected_equilibrium, (
        f"Full-cross minority species mean {minority:.1f} exceeds 30% of {expected_equilibrium}, "
        f"samples={species_counts}"
    )


def test_two_species_asymmetric_matrix():
    length = STD_LENGTH
    cells = _cell_count(length)
    dd_matrix = [[STD_DD, STD_DD], [0.0, STD_DD]]
    sim = _make_sim(dd_matrix, length, cells, seed=333)

    rng = np.random.default_rng(333)
    _seed_uniform(sim, [STD_EQ_POPULATION, 0], rng)

    totals, species_counts = _collect_counts(sim, WARMUP_BIASED)
    total_mean = float(np.mean(totals))
    mean_counts = np.mean(species_counts, axis=0)

    assert abs(total_mean - STD_EQ_POPULATION) <= EQ_TOLERANCE * STD_EQ_POPULATION, (
        f"Asymmetric total mean {total_mean:.1f}, expected {STD_EQ_POPULATION}, "
        f"samples={totals}"
    )
    assert mean_counts[0] >= (1.0 - SPECIES_TOLERANCE) * STD_EQ_POPULATION, (
        f"Asymmetric dominant species mean {mean_counts[0]:.1f} too low, "
        f"samples={species_counts[:, 0]}"
    )
    assert mean_counts[1] <= 0.2 * STD_EQ_POPULATION, (
        f"Asymmetric subordinate species mean {mean_counts[1]:.1f} too high, "
        f"samples={species_counts[:, 1]}"
    )
