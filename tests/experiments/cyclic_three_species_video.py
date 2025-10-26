"""
Generate a 2D cyclic three-species SSA animation with rock-paper-scissors interactions.

The experiment uses the standard birth/death kernels from `tests.test_equal_kernels_nd`
and the interaction pattern specified by the user:
    species 1 wins over species 2
    species 2 wins over species 3
    species 3 wins over species 1

The resulting MP4 video is saved next to this script.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SSA import make_ssa_state_2d
from tests.test_equal_kernels_nd import (
    STD_B,
    STD_CUTOFF,
    STD_D,
    STD_DD,
    half_normal_equal_tables_2d,
)

# Simulation configuration
SPECIES_COUNT = 3
DOMAIN_SIZE = np.array([50.0, 50.0], dtype=np.float64)
CELL_COUNT = np.array([50, 50], dtype=np.int32)
TIME_STEP = 1.0
TOTAL_TIME = 500.0
INITIAL_PER_SPECIES = 10000
SEED = 1729

# Output
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "cyclic_three_species.mp4"

# Visualization - Using plasma colormap for better contrast
POINT_SIZE = 8.0
POINT_ALPHA = 0.75
FPS = 15
BACKGROUND_COLOR = "#F5F5F5"


def _prepare_kernels(species_count: int):
    """Duplicate the reference kernels for every species/species pair."""
    birth_x, birth_y, death_x, death_y = half_normal_equal_tables_2d(cutoff=STD_CUTOFF)

    birth_x = np.repeat(birth_x, species_count, axis=0)
    birth_y = np.repeat(birth_y, species_count, axis=0)

    death_x = np.repeat(death_x, species_count, axis=0)
    death_x = np.repeat(death_x, species_count, axis=1)
    death_y = np.repeat(death_y, species_count, axis=0)
    death_y = np.repeat(death_y, species_count, axis=1)

    cutoffs = np.full((species_count, species_count), STD_CUTOFF, dtype=np.float64)
    return birth_x, birth_y, death_x, death_y, cutoffs


def _prepare_dd_matrix() -> np.ndarray:
    """
    Construct the cyclic dominance matrix.
    """
    dominance_matrix = np.array(
        [
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    dd_matrix = STD_DD * dominance_matrix.T
    return dd_matrix


def build_simulation():
    birth_x, birth_y, death_x, death_y, cutoffs = _prepare_kernels(SPECIES_COUNT)
    dd_matrix = _prepare_dd_matrix()

    sim = make_ssa_state_2d(
        M=SPECIES_COUNT,
        area_len=DOMAIN_SIZE,
        cell_count=CELL_COUNT,
        birth_rates=np.full(SPECIES_COUNT, STD_B, dtype=np.float64),
        death_rates=np.full(SPECIES_COUNT, STD_D, dtype=np.float64),
        dd_matrix=dd_matrix,
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=cutoffs,
        seed=SEED,
        is_periodic=True,
    )

    rng = np.random.default_rng(SEED)
    for species_id in range(SPECIES_COUNT):
        xs = rng.uniform(0.0, DOMAIN_SIZE[0], size=INITIAL_PER_SPECIES)
        ys = rng.uniform(0.0, DOMAIN_SIZE[1], size=INITIAL_PER_SPECIES)
        for x, y in zip(xs, ys, strict=False):
            if not sim.spawn_particle(species_id, float(x), float(y)):
                raise RuntimeError(
                    "Failed to seed initial population. This should not happen with dynamic capacity."
                )

    return sim


class SimulationFrameIterator:
    """Iterate SSA frames by advancing the simulation one time step per frame."""

    def __init__(self, sim, total_steps: int):
        self.sim = sim
        self.total_steps = total_steps
        self._step = 0

    def __iter__(self) -> "SimulationFrameIterator":
        return self

    def __next__(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        if self._step > self.total_steps:
            raise StopIteration

        if self._step > 0:
            self.sim.run_until_time(TIME_STEP)

        total = self.sim.current_population()
        if total > 0:
            xs = np.copy(self.sim.positions[:total, 0])
            ys = np.copy(self.sim.positions[:total, 1])
            species_ids = np.copy(self.sim.species_id[:total])
        else:
            xs = np.empty(0, dtype=np.float64)
            ys = np.empty(0, dtype=np.float64)
            species_ids = np.empty(0, dtype=np.int32)

        if self._step % 100 == 0:
            print(f"Simulated frame {self._step:04d}/{self.total_steps} | population={total}")

        payload = (self._step, xs, ys, species_ids)
        self._step += 1
        return payload


def _ensure_ffmpeg_writer() -> None:
    if animation.writers.is_available("ffmpeg"):
        return

    if imageio_ffmpeg is not None:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Unable to obtain ffmpeg binary from imageio_ffmpeg ({exc}). "
                "Falling back to default matplotlib configuration.",
                RuntimeWarning,
            )
        else:
            plt.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
            if animation.writers.is_available("ffmpeg"):
                return

    raise RuntimeError(
        "Matplotlib could not find an ffmpeg binary to encode MP4 output. "
        "Install ffmpeg and ensure it is on your PATH, or install the 'imageio-ffmpeg' "
        "package which bundles a binary."
    )


def create_animation(sim, output_path: Path):
    _ensure_ffmpeg_writer()
    total_steps = int(TOTAL_TIME / TIME_STEP)
    frame_iter = SimulationFrameIterator(sim, total_steps)

    plt.ioff()
    # Improved figure with better styling
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    fig.patch.set_facecolor("white")
    
    # Use plasma colormap for vibrant colors
    norm = plt.Normalize(vmin=0, vmax=SPECIES_COUNT - 1)

    scatter = ax.scatter(
        [], [],
        s=POINT_SIZE,
        c=[],
        cmap='plasma',
        norm=norm,
        edgecolors="none",
        alpha=POINT_ALPHA,
    )
    scatter.set_clim(0, SPECIES_COUNT - 1)

    # Enhanced time display with larger font and better visibility
    time_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#1A1A1A",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor="#333333",
            linewidth=1.5
        ),
    )
    
    # Population counter
    pop_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#1A1A1A",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.85,
            edgecolor="#666666",
            linewidth=1
        ),
    )

    ax.set_xlim(0.0, DOMAIN_SIZE[0])
    ax.set_ylim(0.0, DOMAIN_SIZE[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Improved labels and title with better styling
    ax.set_xlabel("Position X", fontsize=12, fontweight="bold")
    ax.set_ylabel("Position Y", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cyclic Competition: Species 1 → 2 → 3 → 1",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    
    # Add subtle grid for better readability
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5, color="#666666")
    ax.set_axisbelow(True)
    
    # Improve tick styling
    ax.tick_params(axis="both", which="major", labelsize=10, length=5, width=1.2)
    
    # Add colorbar to show species mapping
    plasma_cmap = cm.get_cmap('plasma')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plasma_cmap(0.0), edgecolor="black", label="Species 1", alpha=POINT_ALPHA),
        Patch(facecolor=plasma_cmap(0.5), edgecolor="black", label="Species 2", alpha=POINT_ALPHA),
        Patch(facecolor=plasma_cmap(1.0), edgecolor="black", label="Species 3", alpha=POINT_ALPHA),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        framealpha=0.9,
        edgecolor="#333333",
        fontsize=10,
        bbox_to_anchor=(0.02, 0.92)
    )

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([], dtype=np.float64))
        time_text.set_text("Time: 0.0")
        pop_text.set_text("Population: 0")
        return scatter, time_text, pop_text

    def update(frame_data):
        step_idx, xs, ys, species_ids = frame_data
        total_pop = xs.size
        
        if total_pop == 0:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([], dtype=np.float64))
        else:
            coords = np.column_stack((xs, ys))
            scatter.set_offsets(coords)
            scatter.set_array(species_ids.astype(np.float64))
        
        # Update time display with better formatting
        current_time = step_idx * TIME_STEP
        time_text.set_text(f"Time: {current_time:.1f}")
        
        # Update population count
        pop_text.set_text(f"Pop: {total_pop:,}")
        
        return scatter, time_text, pop_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_iter,
        init_func=init,
        blit=False,
        repeat=False,
        save_count=total_steps + 1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving animation to {output_path} (this may take a while)...")
    anim.save(str(output_path), writer="ffmpeg", fps=FPS)
    plt.close(fig)
    print(f"Done. Animation saved to {output_path}")


def main():
    sim = build_simulation()
    create_animation(sim, OUTPUT_PATH)


if __name__ == "__main__":
    main()
