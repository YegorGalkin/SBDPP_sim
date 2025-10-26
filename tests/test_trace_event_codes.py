import numpy as np

from SSA import (
    make_ssa_state_1d,
    EVENT_MANUAL_SPAWN,
)
from tests.test_equal_kernels_nd import half_normal_equal_tables_1d, STD_CUTOFF, STD_B, STD_D, STD_DD


def test_manual_spawn_uses_manual_event_code():
    length = 50.0
    cells = 256
    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_1d()
    state = make_ssa_state_1d(
        M=1,
        area_len=length,
        cell_count=cells,
        birth_rates=np.array([STD_B]),
        death_rates=np.array([STD_D]),
        dd_matrix=np.array([[STD_DD]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[STD_CUTOFF]]),
        seed=42,
        cell_capacity=512,
        is_periodic=False,
        trace_enabled=True,
        trace_capacity=8,
    )

    ok = state.spawn_particle_traced(0, 0.5 * length)
    assert ok is True
    assert int(state.trace_count) == 1
    assert int(state.trace_type[0]) == EVENT_MANUAL_SPAWN
