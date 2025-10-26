# Front Speed Measurement Experiment

This experiment measures the propagation speed of a spatial birth-death point process in 1D, 2D, and 3D using the SSA simulators.

## Running the experiment

```bash
python tests/experiments/measure_front_speed.py
```

The script will:
- advance each simulator in fixed time increments (`TIME_STEP`, default 1.0);
- stop once any population gets within the configured kernel cutoff distance of the domain boundary (the final step is discarded);
- estimate front speeds via `scipy.stats.linregress`.

## Outputs

Generated plots are written to `tests/experiments/front_speed_results/`:

| File | Description |
| --- | --- |
| `1d_max_distance.png` | Max front distance vs. time (1D) with linear fit. |
| `1d_left_right_distance.png` | Left/right front distances vs. time (1D) with linear fits. |
| `2d_max_distance.png` | Max front distance vs. time (2D) with linear fit. |
| `2d_border_polygons.png` | Time-stamped polar polygons of the 2D population border (30° bins). |
| `3d_max_distance.png` | Max front distance vs. time (3D) with linear fit. |

Console output lists the fitted slopes (distance per unit time) for each view.

## Dependencies

Ensure the project dependencies (including `matplotlib`, `numpy`, and `scipy`) are installed, e.g.:

```bash
pip install -r requirements.txt
```

Adjust parameters (domain size, time step, angular resolution) by editing `tests/experiments/measure_front_speed.py`.
