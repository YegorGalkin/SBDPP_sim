```markdown
# Spatial Birth-Death Simulator (Cython Wrapper)

This repository contains a C++/Cython implementation of an N-dimensional birth-death point process simulator with spatially explicit interactions. You can build it as a Python extension module, then import and use it from Python to:

- Create grids in 1, 2, or 3 dimensions
- Specify birth/death rates and radial kernels
- Place initial populations
- Run stochastic events (either random events or user-specified spawns/kills)
- Inspect cell-level data (coordinates, death rates, etc.)

---

## 1. Directory Layout

Directory structure for this project looks like:

```
SBDPP_sim/
├── examples/
├── include/
│   └── SpatialBirthDeath.h
├── simulation/
│   ├── __init__.py
│   ├── SpatialBirthDeath.cpp
│   └── SpatialBirthDeathWrapper.pyx
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

- **`include/SpatialBirthDeath.h`**: C++ header for the simulator.
- **`simulation/SpatialBirthDeath.cpp`**: C++ source file with core logic.
- **`simulation/SpatialBirthDeathWrapper.pyx`**: Cython interface to the C++ code.
- **`setup.py`**: Build script using setuptools/Cython.
- **`examples/`**: (Optional) example scripts or notebooks.
- **`requirements.txt`**: Python packages required (e.g., `cython`).
- **`README.md`**: This documentation.

---

## 2. Building the Extension

From the top-level directory (`SBDPP_sim/`), ensure you have a C++20 compiler and Cython installed, then run:

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

This compiles and generates a platform-specific shared library (e.g., `.so` or `.pyd`) in the `simulation/` directory.  
If you change any `.cpp` or `.pyx` files, re-run `python setup.py build_ext --inplace` to rebuild.

---

## 3. Basic Usage from Python

Once built, you can import the extension:

```python
import simulation  # or: from simulation import PyGrid1, PyGrid2, PyGrid3
```

Three Python classes are provided:

- **`PyGrid1`**: 1D simulator
- **`PyGrid2`**: 2D simulator
- **`PyGrid3`**: 3D simulator

They share similar methods and attributes, differing mainly by coordinate dimensionality.

### 3.1 Creating a Grid

A constructor for `PyGrid2` might look like:

```python
import simulation

g2 = simulation.PyGrid2(
    M=2,
    areaLen=[100.0, 80.0],   # domain size in x,y
    cellCount=[10, 8],       # 10 x 8 cells
    isPeriodic=False,
    birthRates=[0.5, 0.2],   # baseline birth rates
    deathRates=[0.1, 0.05],  # baseline death rates
    ddMatrix=[0.01, 0.02,    # Flattened MxM matrix
              0.02, 0.01],
    birthX=[[0.0, 1.0], [0.0, 1.0]],
    birthY=[[0.0, 3.0], [0.0, 2.0]],
    deathX_=[
      [ [0.0,5.0],[0.0,5.0] ],   # s1=0 -> [ s2=0, s2=1 ]
      [ [0.0,5.0],[0.0,5.0] ]    # s1=1 -> [ s2=0, s2=1 ]
    ],
    deathY_=[
      [ [1.0,0.0],[2.0,0.0] ],
      [ [2.0,0.0],[1.0,0.0] ]
    ],
    cutoffs=[5.0, 5.0, 5.0, 5.0],  # Flattened MxM
    seed=42,
    rtimeLimit=3600.0
)
```

**Key parameters**:

1. **M** (int): Number of species.
2. **areaLen** (list of floats): Domain length in each dimension.
3. **cellCount** (list of ints): Number of cells along each dimension.
4. **isPeriodic** (bool): Whether the domain wraps around.
5. **birthRates** (length `M`): Baseline birth rates.
6. **deathRates** (length `M`): Baseline death rates.
7. **ddMatrix** (length `M*M`): Flattened pairwise competition matrix.
8. **birthX**, **birthY** (list-of-lists of floats): Radial birth kernel data per species.
9. **deathX_**, **deathY_** (3-level nested lists): Radial death kernel for each `(s1, s2)`.
10. **cutoffs** (length `M*M`): Cutoff distances for each `(s1, s2)`.
11. **seed** (int): Random number seed.
12. **rtimeLimit** (float): Max real-time limit for certain run methods.

### 3.2 Placing Initial Populations

Use `placePopulation(...)`, which takes a list of lists of coordinates:

- For 1D, each coordinate is `[x]`.
- For 2D, each coordinate is `[x, y]`.
- For 3D, each coordinate is `[x, y, z]`.

Example for 2D, with 2 species:

```python
initCoords = [
    [ [10.0, 20.0], [30.0, 40.0] ],  # species 0
    [ [ 5.0,  2.0], [ 7.0,  9.0] ]   # species 1
]
g2.placePopulation(initCoords)
```

### 3.3 Running Events

- **`make_event()`**: Perform one stochastically chosen birth or death event.
- **`spawn_random()`**: Force one random birth event.
- **`kill_random()`**: Force one random death event.
- **`run_events(n)`**: Perform `n` events sequentially (unless rates or real-time limit stop it).
- **`run_for(t)`**: Continue events until simulation time advances by `t` or real-time limit is reached.

Example:

```python
print("Population before events:", g2.total_population)
g2.run_events(1000)
print("After 1000 events, time =", g2.time, ", population =", g2.total_population)

g2.run_for(10.0)
print("After run_for(10.0), time =", g2.time)
```

### 3.4 Cell-Level Data

Each grid cell can be queried via:

- **`get_cell_coords(cell_index, species_idx)`**: Returns a list of coordinates for that species in that cell.
- **`get_cell_death_rates(cell_index, species_idx)`**: Returns a list of per-individual death rates.
- **`get_cell_population(cell_index)`**: Returns a list with the population of each species in that cell.
- **`get_cell_birth_rate(cell_index)`** and **`get_cell_death_rate(cell_index)`**: Returns aggregate rates for that cell.

Cells are indexed `0` to `total_num_cells - 1`. In 2D or 3D, the indexing is flattened. See the C++ `flattenIdx` method for details.

### 3.5 Retrieving All Particle Coordinates

A convenient method **`get_all_particle_coords()`** returns all particle positions in one call, grouped by species:

- **`PyGrid1`** returns `[ [x1, x2, ...], [x1, x2, ...], ... ]`.
- **`PyGrid2`** returns `[ [[x1,y1],[x2,y2],...], [[x1,y1],[x2,y2],...], ... ]`.
- **`PyGrid3`** returns `[ [[x1,y1,z1],[x2,y2,z2],...], [...], ... ]`.

Example:

```python
all_coords = g2.get_all_particle_coords()
for s_idx, coords_list in enumerate(all_coords):
    print(f"Species {s_idx} has {len(coords_list)} particles.")
```

---

## 4. Defining Birth and Death Kernels

The simulator uses **radially symmetric** kernels for both births and deaths, specified as piecewise functions `(X, Y)` that are linearly interpolated at runtime.

- **Birth kernels** allow the simulator to sample a distance from the parent to place a new individual.  
- **Death kernels** define how the presence of a neighbor at distance `r` modifies an individual’s death rate.

### 4.1 Birth Kernels (Inverse Radial CDF)

For each species `s`, you pass `(birthX[s], birthY[s])` representing an **inverse radial CDF** from `[0..1]` to `[0..∞)`. Specifically:

1. **`birthX[s]`** is a sorted array of quantiles in `[0..1)`.  
2. **`birthY[s]`** is the corresponding array of radii, i.e. `ICDF(u)`.

When a birth event occurs for species `s`:
1. A uniform random `u ∈ [0,1]` is drawn.  
2. The radius `r` is found by linear interpolation of `(birthX[s], birthY[s])`.  
3. **1D case**: That radius is multiplied by a random sign (+1 or -1) so the new individual may appear left or right. Since `birthY[s]` only stores nonnegative values, the sign is handled by the simulator itself.  
4. **2D/3D case**: That radius is multiplied by a random direction in 2D (angle) or in 3D (two angles).

#### 1D Example (Half-Normal with Parameter σ)

Using `scipy.stats.halfnorm` for a half-normal distribution with scale=σ (equivalent to `|Normal(0,σ)|`):

```python
import numpy as np
from scipy.stats import halfnorm

sigma = 1.0
N = 1001
epsilon = 1e-3  # to avoid potential infinite tail at u=1
uvals = np.linspace(0, 1 - epsilon, N)
rvals = halfnorm.ppf(uvals, scale=sigma)

birthX_1d = [uvals.tolist()]  # single species
birthY_1d = [rvals.tolist()]
```

#### 2D Example (Rayleigh for Standard Normal, Parameter σ)

A 2D standard normal (with each axis ~ N(0,σ^2)) has a Rayleigh radial distribution, `rayleigh(scale=σ)`:

```python
import numpy as np
from scipy.stats import rayleigh

sigma = 1.0
N = 1001
epsilon = 1e-3
uvals = np.linspace(0, 1 - epsilon, N)
rvals = rayleigh.ppf(uvals, scale=sigma)

birthX_2d = [uvals.tolist()]
birthY_2d = [rvals.tolist()]
```

#### 3D Example (Maxwell-Boltzmann for Standard Normal, Parameter σ)

A 3D standard normal leads to a Maxwell–Boltzmann radial distribution, `maxwell(scale=σ)`:

```python
import numpy as np
from scipy.stats import maxwell

sigma = 1.0
N = 1001
epsilon = 1e-3
uvals = np.linspace(0, 1 - epsilon, N)
rvals = maxwell.ppf(uvals, scale=sigma)

birthX_3d = [uvals.tolist()]
birthY_3d = [rvals.tolist()]
```

---

### 4.2 Death Kernels (Normalized in 1D, 2D, 3D)

For each pair `(s1, s2)`, you must pass `(deathX[s1][s2], deathY[s1][s2])`, plus a **cutoff** in `cutoffs[s1*M + s2]`. If two individuals of species `s1` and `s2` are at distance `r` (within the cutoff), the contribution to occupant j’s death rate is `dd[s1][s2] * kernel(r)` (interpolated from `(X, Y)`).

**Important**: The model assumes these kernels are **normalized** when integrated with respect to the dimension’s volume element:

```
1D:  ∫(0..∞) K(r) dr = 1
2D:  ∫(0..∞) 2π r K(r) dr = 1
3D:  ∫(0..∞) 4π r^2 K(r) dr = 1
```

Such normalization ensures that if `dd[s1][s2] = 1`, the total “average” effect is 1 over the domain (or up to the cutoff).

#### Example: 2D Standard Normal Kernel

For a 2D standard normal with std dev σ, one valid radial factor is:

```
K(r) = (1 / σ^2) * exp( -r^2 / (2σ^2) )
```

so that

```
∫(0..∞) 2π r * K(r) dr = 1.
```

Discretize up to some `max_r` (the cutoff):

```python
import numpy as np

def normal_2d_kernel(r, sigma=1.0):
    return (1.0 / (sigma**2)) * np.exp(-0.5*(r**2)/(sigma**2))

max_r = 5.0
N = 501
distances = np.linspace(0, max_r, N)
values = [normal_2d_kernel(r, sigma=1.0) for r in distances]

deathX_2d = [[ distances.tolist() ]]  # if M=1 (one species)
deathY_2d = [[ values ]]
cutoffs = [max_r]
```

#### Example: 3D Standard Normal Kernel

For a 3D standard normal with std dev σ, you want

```
∫(0..∞) 4π r^2 K(r) dr = 1.
```

A possible radial factor is:

```
K(r) = (constant) * exp( - r^2 / (2σ^2) ) * (r^2 / σ^2)
```

where the constant is chosen so the integral is 1. For instance:

```python
import numpy as np

def normal_3d_kernel(r, sigma=1.0):
    # Approximates the correct factor for 3D normalization
    c = 4.0 * np.pi / ((2.0 * np.pi)**1.5 * sigma**3)
    return c * np.exp(-0.5*(r**2)/(sigma**2)) * (r**2 / sigma**2)

max_r = 5.0
N = 501
distances = np.linspace(0, max_r, N)
values = [normal_3d_kernel(r, sigma=1.0) for r in distances]

deathX_3d = [[ distances.tolist() ]]
deathY_3d = [[ values ]]
cutoffs = [max_r]
```

---

## 5. Example Usage

A minimal usage example in 2D with no competition might look like:

```python
import simulation

g2 = simulation.PyGrid2(
    M=1,
    areaLen=[25.0, 25.0],
    cellCount=[25, 25],
    isPeriodic=True,
    birthRates=[1.0],
    deathRates=[0.0],
    ddMatrix=[0.0],  # single-species => 1x1
    # trivial birth kernel from 0..1 -> radius 0..2
    birthX=[[0.0, 1.0]],
    birthY=[[0.0, 2.0]],
    # no competition => death kernel always 0
    deathX_=[ [ [0.0, 5.0] ] ],
    deathY_=[ [ [0.0, 0.0] ] ],
    cutoffs=[5.0],
    seed=42,
    rtimeLimit=3600.0
)

initCoords = [
  [ [2.0, 5.0], [10.0, 10.0], [20.0, 8.0] ]
]
g2.placePopulation(initCoords)

print("Initial pop:", g2.total_population)
for i in range(5):
    g2.make_event()
    print(f"After event {i+1}, time={g2.time:.3f}, pop={g2.total_population}")

allcoords = g2.get_all_particle_coords()
print("All coords (2D):", allcoords)
```

---

## 6. License & Contributions

- **License**: [Your License Here], see `LICENSE.txt`.
- Contributions & bug reports are welcome.  
- Please open issues or pull requests to improve this simulator!

Enjoy simulating your spatial birth‐death processes!
```
