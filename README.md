Below is an example **README.md** that explains how to install the extension, create instances of the simulator, and interact with it. Feel free to adapt to your exact directory structure and usage patterns.

---

# Spatial Birth-Death Simulator (Cython Wrapper)

This directory provides a C++/Cython implementation of an N-dimensional birth-death process simulator with spatially explicit interactions. You can build it as a Python extension module, then import and use it from Python to:

- Create grids in 1, 2, or 3 dimensions
- Specify birth/death rates and radial kernels
- Place initial populations
- Run stochastic events (either random events or user-specified spawns/kills)
- Inspect cell-level data (coordinates, death rates, etc.)

---

## 1. Building the Extension

### Directory Layout

Suppose your directory looks like this:

```
my_sim_project/
├── include/
│    └── SpatialBirthDeath.h        # C++ header
├── simulation/
│    ├── SpatialBirthDeath.cpp      # C++ source
│    ├── SpatialBirthDeathWrapper.pyx
│    └── __init__.py
├── setup.py
└── README.md   # <--- this file
```

The `setup.py` script uses `Cython.Build.cythonize` to compile and link the `.pyx` and `.cpp` files into a Python extension. 

**Build and install**:

```bash
cd my_sim_project
python setup.py build_ext --inplace
```

This compiles your code to a shared library (e.g. `SpatialBirthDeath.*.so` on Linux or `SpatialBirthDeath.*.pyd` on Windows) inside the `simulation` folder.

> **Tip**: If you change the C++ or .pyx files, you must re-run the `setup.py build_ext --inplace` command to recompile.

---

## 2. Basic Usage from Python

Once built, you can import the extension module in Python sessions or scripts:

```python
import simulation  # or from simulation import PyGrid1, PyGrid2, PyGrid3
```

Inside `simulation/SpatialBirthDeathWrapper.pyx`, there are three Python classes:

- **`PyGrid1`**: For a 1D simulator  
- **`PyGrid2`**: For a 2D simulator  
- **`PyGrid3`**: For a 3D simulator  

They share the same interface but differ in how the spatial coordinates are represented (1D vs. 2D vs. 3D).

### 2.1 Creating a Grid

When creating a `PyGrid2`, for example, the constructor expects:

1. `M`: number of species (int)  
2. `areaLen`: list of dimension sizes. For 2D, `[width, height]`.  
3. `cellCount`: number of cells along each dimension. For 2D, `[nx, ny]`.  
4. `isPeriodic`: bool indicating whether edges wrap around.  
5. `birthRates`: Python list of length `M`, the baseline birth rates for each species.  
6. `deathRates`: Python list of length `M`, the baseline death rates for each species.  
7. `ddMatrix`: Flattened MxM matrix of pairwise competition coefficients.  
8. `birthX`: A list of length `M`, each an array of the **x-values** of the birth kernel’s CDF or ICDF.  
9. `birthY`: A list of length `M`, each an array of the corresponding **y-values**.  
10. `deathX_`: A list of lists of lists of floats. For each pair `(s1, s2)`, you provide a radial kernel x-values for the competition function.  
11. `deathY_`: The corresponding radial kernel y-values.  
12. `cutoffs`: A flattened MxM list of distances beyond which competition is ignored, one for each `(s1, s2)`.  
13. `seed`: Integer seed for the random number generator.  
14. `rtimeLimit`: Maximum wall-clock time in seconds (used by some run methods).

For example:

```python
import simulation

g2 = simulation.PyGrid2(
    M=2,
    areaLen=[100.0, 80.0],     # 2D domain 100 wide x 80 tall
    cellCount=[10, 8],         # 10 x 8 grid of cells
    isPeriodic=False,
    birthRates=[0.5, 0.2],     # per-species birth rates
    deathRates=[0.1, 0.05],    # per-species death rates
    ddMatrix=[0.01, 0.02,      # Flattened 2x2
              0.02, 0.01],    
    birthX=[[0.0,1.0],[0.0,1.0]],     # e.g. trivial placeholder
    birthY=[[0.0,3.0],[0.0,2.0]],
    deathX_=[ [ [0.0,5.0],[0.0,5.0] ],  # For s1=0 -> [ [x-values for s2=0], [x-values for s2=1] ]
              [ [0.0,5.0],[0.0,5.0] ] ],
    deathY_=[ [ [1.0,0.0],[2.0,0.0] ],
              [ [2.0,0.0],[1.0,0.0] ] ],
    cutoffs=[5.0, 5.0,
             5.0, 5.0],
    seed=42,
    rtimeLimit=3600.0
)
```

### 2.2 Placing Initial Populations

Each grid type has a `placePopulation(...)` method to set initial coordinates of individuals. The argument is a **list of lists**:

- For 1D (`PyGrid1`): `initCoords[s]` is a list of 1D positions, each `[x]`.  
- For 2D (`PyGrid2`): `initCoords[s]` is a list of 2D positions, each `[x, y]`.  
- For 3D (`PyGrid3`): `initCoords[s]` is a list of 3D positions, each `[x, y, z]`.

Example for 2D and `M=1`:

```python
initCoords = [
  [ [10.0, 20.0], [30.0, 40.0] ]
]
g2.placePopulation(initCoords)
```

Here, `initCoords[0]` is a list of two positions for species 0 (because we have only 1 species). If `M=2`, you’d have two sub-lists, one per species.

### 2.3 Running Events

You have multiple ways to run or advance the simulation:

- **`make_event()`**: Executes exactly one birth/death event (chosen stochastically based on current total rates).  
- **`spawn_random()`**: Forcibly do one birth event (randomly chosen location/species).  
- **`kill_random()`**: Forcibly do one death event (random).  
- **`run_events(n)`**: Repeatedly calls `make_event()` `n` times, stopping early if the real-time limit is reached.  
- **`run_for(t)`**: Calls events until simulation time increases by `t`, or the real-time limit is reached.

Example:

```python
print("Before, total_population =", g2.total_population)
g2.make_event()
print("After 1 event, time =", g2.time, "pop =", g2.total_population)

# run for 1000 events
g2.run_events(1000)
print("After 1000 events, time =", g2.time)

# run for 10 more time units
g2.run_for(10.0)
print("After run_for(10.0), sim time =", g2.time)
```

### 2.4 Reading Cell Data

Each grid has `get_cell_coords(cell_index, species_idx)`, `get_cell_death_rates(cell_index, species_idx)`, etc.:

- **`get_cell_coords(cell_index, species_idx)`**  
  Returns a list of coordinate arrays. For 2D, each element is `[x, y]`. For 1D, each is a single float.  
- **`get_cell_death_rates(cell_index, species_idx)`**  
  Returns a list of floating‐point death rates, one per individual.  
- **`get_cell_population(cell_index)`**  
  Returns a list of population counts, one entry per species.  
- **`get_cell_birth_rate(cell_index)`** and **`get_cell_death_rate(cell_index)`**  
  Returns the aggregate birth/death rate of that cell (sum over individuals).  

The **`cell_index`** is an integer from 0 up to `(total_num_cells - 1)`. For 2D, cells are flattened in row-major or column-major order (depending on how the C++ code is written). You can check the docstrings or see how `Grid<2>::flattenIdx` is computed in your C++ code.

Example usage:

```python
ncells = g2.get_num_cells()
for c in range(ncells):
    coords = g2.get_cell_coords(c, 0)  # species 0
    print(f"Cell {c}: coords of species 0:", coords)

    poplist = g2.get_cell_population(c)
    print(f"  population by species = {poplist}")

    cell_drate = g2.get_cell_death_rate(c)
    print(f"  cell-level death rate = {cell_drate}")
```

---

## 3. Example Usage

Below is a minimal usage example in 2D:

```python
import simulation
import numpy as np

g2 = simulation.PyGrid2(
    M=1,
    areaLen=[25.0, 25.0],
    cellCount=[25, 25],
    isPeriodic=True,
    birthRates=[1.0],
    deathRates=[0.0],
    ddMatrix=[0.0],  # single-species, no competition
    birthX=[[0.0,1.0]],  # trivial CDF domain
    birthY=[[0.0,2.0]],  # trivial "ICDF" or kernel
    deathX_=[ [ [0.0, 1.0] ] ],  # single species
    deathY_=[ [ [1.0, 0.0] ] ],
    cutoffs=[1.0],
    seed=42,
    rtimeLimit=3600.0
)

# Place 3 initial individuals at different spots
initCoords = [
  [ [2.0, 5.0], [10.0,10.0], [20.0, 8.0] ]
]
g2.placePopulation(initCoords)

print("Initial pop:", g2.total_population)  # 3

for i in range(5):
    g2.make_event()
    print(f"After event {i+1}, time={g2.time:.3f}, population={g2.total_population}")

# Inspect cell-level data
nc = g2.get_num_cells()
print(f"Total cells = {nc}")
some_coords = g2.get_cell_coords(0, 0)
print("Species 0 coords in cell 0:", some_coords)
```

---

## 4. Notes on the Nested List Formats

1. **`birthX` and `birthY`** are each lists of length `M`.  
   - `birthX[s]` is a sorted array of x-values (often the domain of the CDF or an inverse-CDF).  
   - `birthY[s]` is the corresponding array of y-values.  

2. **`deathX_`** is a 3-level structure: for each `(s1, s2)` pair, you have an array of radial distances. So if you have `M=2`, `deathX_` must be length 2 (outer), each of which is length 2 (middle), each of which is a list of floats.  
   - E.g. `deathX_[s1][s2] = [0.0, 2.0, 5.0]` might be the radial grid.  

3. **`deathY_`** parallels `deathX_`. For each `(s1, s2)` pair, `deathY_[s1][s2]` is an array of float kernel values or partial sums.

4. **`cutoffs`** is a flattened MxM list of max distances. For instance, if `M=3`, you have 9 numbers. Each number is the cutoff beyond which interactions are ignored for a given `(s1, s2)`.

5. **Initial coordinates** (for `placePopulation`) must be:

   - 1D: `initCoords[s][i] = [x_i]`
   - 2D: `initCoords[s][i] = [x_i, y_i]`
   - 3D: `initCoords[s][i] = [x_i, y_i, z_i]`

   Where `s` indexes the species.

---

## 5. Known Caveats / Tips

1. **Boundary conditions**: 
   - If `isPeriodic=True`, coordinates that go “off the edge” will wrap around.  
   - If `False`, any newborn that falls out of the domain is discarded.  

2. **Discrete Distributions**:  
   - If your total birth or death rate becomes zero, calling `spawn_random()` or `kill_random()` might lead to undefined behavior in `std::discrete_distribution`. The code typically checks if rates are near zero and returns immediately, but be aware of edge cases.  

3. **Flattened Cell Indices**:  
   - For 2D, the flattening is likely `idx = i + j * cellCount[0]`, or some variation. See `flattenIdx(...)` in the C++ code if you want to convert `(i,j)` to a single integer.  

4. **Performance**:  
   - For large grids, copying data out of `get_cell_coords(...)` for every cell can be slow. You might prefer aggregated data or a more direct NumPy approach.

5. **Thread‐safety**:  
   - Not guaranteed. Typically you use one `PyGridX` in a single Python thread at a time.  

---

## 6. License & Contributions

- The simulator is under [Your License Here], see `LICENSE.txt`.  
- Contributions & bug reports welcome.  

Please feel free to reach out if you have questions or suggestions. Enjoy simulating spatial birth‐death processes!