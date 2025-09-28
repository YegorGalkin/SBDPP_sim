import numpy as np

from SSA import (
    make_ssa_state_1d,
    alive_indices,
    get_all_particle_coords,
    get_all_particle_death_rates,
)


def make_state(seed: int = 123):
    M = 2
    area_len = 10.0
    cell_count = 10

    birth_rates = np.array([0.4, 0.3], dtype=np.float64)
    death_rates = np.array([0.2, 0.1], dtype=np.float64)
    dd_matrix = np.zeros((M, M), dtype=np.float64)

    birth_x = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    birth_y = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float64)

    single_axis = np.array([0.0, 1.0], dtype=np.float64)
    death_x = np.broadcast_to(single_axis, (M, M, single_axis.size)).copy()
    death_y = np.zeros((M, M, single_axis.size), dtype=np.float64)

    cutoffs = np.zeros((M, M), dtype=np.float64)

    return make_ssa_state_1d(
        M=M,
        area_len=area_len,
        cell_count=cell_count,
        birth_rates=birth_rates,
        death_rates=death_rates,
        dd_matrix=dd_matrix,
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=cutoffs,
        seed=seed,
        cell_capacity=16,
    )


def test_spawn_population_and_queries():
    state = make_state()
    initial_positions = [
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([6.0], dtype=np.float64),
    ]
    for species_id, arr in enumerate(initial_positions):
        for pos in arr:
            assert state.spawn_particle(species_id, float(pos))

    assert state.current_population() == 3

    coords = get_all_particle_coords(state)
    assert len(coords) == 2
    assert np.allclose(np.sort(coords[0]), np.array([1.0, 2.0]))
    assert np.allclose(np.sort(coords[1]), np.array([6.0]))

    rates = get_all_particle_death_rates(state)
    assert len(rates) == 2
    assert all(isinstance(arr, np.ndarray) for arr in rates)


def test_spawn_and_kill_particle():
    state = make_state()
    assert state.spawn_particle(0, 1.0)
    assert state.spawn_particle(0, 3.5)

    population_after_spawn = state.current_population()
    assert population_after_spawn == 2

    alive = alive_indices(state)
    assert alive.shape[0] == population_after_spawn
    assert state.kill_particle_index(int(alive[0]))
    assert state.current_population() == population_after_spawn - 1


def test_random_events_progression():
    state = make_state(seed=999)
    initial = [
        np.array([1.5, 2.5], dtype=np.float64),
        np.array([7.5], dtype=np.float64),
    ]
    for species_id, arr in enumerate(initial):
        for pos in arr:
            assert state.spawn_particle(species_id, float(pos))

    state.spawn_random()
    state.kill_random()
    performed = state.run_events(5)
    assert performed <= 5
    state.run_until_time(0.5)
    assert state.current_event_count() > 0
    assert state.current_time() >= 0.0


