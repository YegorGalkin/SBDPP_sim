# Wrapper for PyGrid1D
from .grid_1d_interface import PyGrid1D


class PyGrid1DWrapper:

    def __init__(self, area_length_x: float, cell_count_x: int,
                 b: float, d: float, dd: float, seed: int,
                 initial_population: list, death_values: list,
                 death_cutoff_r: float, birth_values: list,
                 periodic: bool, realtime_limit: float):
        """
        Python wrapper for the Cython PyGrid1D class.

        Parameters
        ----------
        area_length_x : float
            Length of the simulation area along the x-axis.
        cell_count_x : int
            Number of cells along the x-axis.
        b : float
            Birth rate parameter.
        d : float
            Death rate parameter.
        dd : float
            Additional death parameter.
        seed : int
            Random seed for reproducibility.
        initial_population : list of float
            Initial population densities for each cell.
        death_values : list of float
            Death interaction function values.
        death_cutoff_r : float
            Maximum range of death interaction.
        birth_values : list of float
            Birth rate inverse cumulative distribution function values.
        periodic : bool
            Whether the simulation area is periodic.
        realtime_limit : float
            Real-time limit for the simulation in seconds.
        """
        self._validate_parameters(area_length_x, cell_count_x, b, d, dd, seed,
                                  initial_population, death_values, death_cutoff_r,
                                  birth_values, periodic, realtime_limit)
        self._grid = PyGrid1D(
            area_length_x, cell_count_x, b, d, dd, seed,
            initial_population, death_values,
            death_cutoff_r, birth_values, periodic, realtime_limit
        )

    def __getattr__(self, name):
        # Delegate attribute access to the underlying Cython object
        return getattr(self._grid, name)

    @staticmethod
    def _validate_parameters(area_length_x, cell_count_x, b, d, dd, seed,
                              initial_population, death_values, death_cutoff_r,
                              birth_values, periodic, realtime_limit):
        if not isinstance(area_length_x, (float, int)) or area_length_x <= 0:
            raise ValueError("area_length_x must be a positive number.")
        if not isinstance(cell_count_x, int) or cell_count_x <= 0:
            raise ValueError("cell_count_x must be a positive integer.")
        if not isinstance(b, (float, int)) or b < 0:
            raise ValueError("b must be a non-negative number.")
        if not isinstance(d, (float, int)) or d < 0:
            raise ValueError("d must be a non-negative number.")
        if not isinstance(dd, (float, int)) or dd < 0:
            raise ValueError("dd must be a non-negative number.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not isinstance(initial_population, list) or not all(isinstance(x, (float, int)) for x in initial_population):
            raise ValueError("initial_population must be a list of numbers.")
        if not isinstance(death_values, list) or not all(isinstance(x, (float, int)) for x in death_values):
            raise ValueError("death_values must be a list of numbers.")
        if not isinstance(death_cutoff_r, (float, int)) or death_cutoff_r <= 0:
            raise ValueError("death_cutoff_r must be a positive number.")
        if not isinstance(birth_values, list) or not all(isinstance(x, (float, int)) for x in birth_values):
            raise ValueError("birth_values must be a list of numbers.")
        if not isinstance(periodic, bool):
            raise ValueError("periodic must be a boolean.")
        if not isinstance(realtime_limit, (float, int)) or realtime_limit <= 0:
            raise ValueError("realtime_limit must be a positive number.")
