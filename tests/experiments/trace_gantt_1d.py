import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SSA import make_ssa_state_1d
from tests.test_equal_kernels_nd import half_normal_equal_tables_1d

sns.set_theme(style="whitegrid")


def run_trace_1d(
    first_n: int = 1000,
    gantt_png: str = "tests/experiments/trace_gantt_1d.png",
):
    palette = sns.color_palette("tab10")
    color_lifeline = palette[0]
    color_birth_marker = palette[2]
    color_death_marker = palette[3]
    color_manual_marker = palette[4]

    b = 1.0
    d = 0.0
    dd = 0.1
    cutoff = 5.0
    L = 200.0
    cells = 400
    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_1d(cutoff)
    sim = make_ssa_state_1d(
        M=1,
        area_len=L,
        cell_count=cells,
        birth_rates=np.array([b]),
        death_rates=np.array([d]),
        dd_matrix=np.array([[dd]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=np.array([[cutoff]]),
        seed=909,
        cell_capacity=4096,
        is_periodic=False,
        trace_enabled=True,
        trace_capacity=first_n * 10,
    )

    center = 0.5 * L
    sim.spawn_particle_traced(0, center)

    while sim.trace_count < first_n and sim.reached_capacity() is False:
        sim.run_events(2000)

    n = min(first_n, int(sim.trace_count))
    times = np.array([sim.trace_time[i] for i in range(n)], dtype=float)
    types = np.array([sim.trace_type[i] for i in range(n)], dtype=int)
    pos = np.array([sim.trace_pos[i, 0] for i in range(n)], dtype=float)

    lifelines: list[dict] = []
    pos_values: list[float] = []
    for t, etype, p in zip(times, types, pos):
        if etype in (0, 2):  # stochastic birth or manual spawn
            lifelines.append(
                {
                    "t_birth": float(t),
                    "t_death": None,
                    "pos_birth": float(p),
                    "pos_death": None,
                    "manual": etype == 2,
                }
            )
            pos_values.append(float(p))
        elif etype == 1:  # death
            open_idx = -1
            best_dist = np.inf
            for idx, rec in enumerate(lifelines):
                if rec["t_death"] is not None:
                    continue
                d = abs(rec["pos_birth"] - p)
                if d < best_dist:
                    best_dist = d
                    open_idx = idx
            if open_idx == -1:
                lifelines.append(
                    {
                        "t_birth": float(t),
                        "t_death": float(t),
                        "pos_birth": float(p),
                        "pos_death": float(p),
                        "manual": False,
                    }
                )
                pos_values.extend([float(p)])
            else:
                lifelines[open_idx]["t_death"] = float(t)
                lifelines[open_idx]["pos_death"] = float(p)
                pos_values.append(float(p))

    if not lifelines:
        print("No events recorded; nothing to plot.")
        return

    t_births = np.array([rec["t_birth"] for rec in lifelines], dtype=float)
    t_end_candidates = [
        rec["t_death"] if rec["t_death"] is not None else float(times.max())
        for rec in lifelines
    ]
    t_min = float(t_births.min())
    t_max = float(max(t_end_candidates))
    if t_max <= t_min:
        t_max = t_min + 1e-6

    pos_array = np.array(pos_values, dtype=float)
    y_min = float(pos_array.min())
    y_max = float(pos_array.max())
    if y_max <= y_min:
        y_max = y_min + 1e-6

    x_lower = t_min - 0.05
    x_upper = t_max + 0.05
    y_lower = y_min - 0.5
    y_upper = y_max + 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    for rec in lifelines:
        y_birth = rec["pos_birth"]
        y_death = rec["pos_death"] if rec["pos_death"] is not None else y_birth
        x0 = rec["t_birth"]
        x1 = rec["t_death"] if rec["t_death"] is not None else t_max

        line_color =  color_lifeline
        ax.plot([x0, x1], [y_birth, y_death], color=line_color, linewidth=1.6, alpha=0.75)

        birth_marker_color = color_manual_marker if rec["manual"] else color_birth_marker
        birth_marker = "*" if rec["manual"] else "o"
        ax.scatter(
            [x0],
            [y_birth],
            c=[birth_marker_color],
            s=70 if rec["manual"] else 32,
            marker=birth_marker,
            alpha=0.95,
            edgecolors="black" if rec["manual"] else "none",
            linewidths=1.0 if rec["manual"] else 0.0,
            zorder=3,
        )

        if rec["t_death"] is not None:
            ax.scatter(
                [x1],
                [y_death],
                c=[color_death_marker],
                s=36,
                marker="x",
                alpha=0.9,
                linewidths=1.8,
                zorder=3,
            )

    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlabel("time")
    ax.set_ylabel("position (1D)")
    ax.set_title(f"1D SSA lifelines (first {n} events)")

    legend_handles = [
        Line2D([0], [0], color=color_lifeline, lw=2, alpha=0.75, label="lifeline"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_birth_marker,
            label="birth",
            linestyle="",
            markeredgecolor="none",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color=color_manual_marker,
            label="manual spawn",
            linestyle="",
            markeredgecolor="black",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=color_death_marker,
            label="death",
            linestyle="",
            markersize=8,
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.9)

    os.makedirs(os.path.dirname(gantt_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(gantt_png, dpi=150)
    plt.close(fig)
    print(f"Saved lifeline plot to: {gantt_png}")


if __name__ == "__main__":
    run_trace_1d()
