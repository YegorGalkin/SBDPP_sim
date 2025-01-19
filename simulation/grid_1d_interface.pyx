# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp cimport bool
cimport cython

cdef extern from "grid_1d.h":
    cdef cppclass Grid1D:
        Grid1D(double area_length_x, int cell_count_x,
               double b, double d, double dd, int seed,
               vector[double] initial_population,
               vector[double] death_values,
               double death_cutoff_r,
               vector[double] birth_values,
               bool periodic, double realtime_limit)

        double area_length_x
        int cell_count_x
        double b
        double d
        double dd
        int seed
        vector[double] initial_population_x
        vector[double] death_y
        double death_cutoff_r
        int death_spline_nodes
        double death_step
        vector[double] birth_inverse_rcdf_y
        int birth_inverse_rcdf_nodes
        double birth_inverse_rcdf_step
        bool periodic
        double realtime_limit
        bool realtime_limit_reached


        void Initialize_death_rates()
        void kill_random()
        void spawn_random()
        void make_event()
        void run_events(int events)
        void run_for(double duration)

        double linear_interpolate(const vector[double]& xgdat, const vector[double]& gdat, double x)
        double death_kernel(double at)
        double birth_inverse_rcdf_kernel(double at)
        double& cell_death_rate_at(int i)
        int& cell_population_at(int i)

        vector[double] get_x_coords_at_cell(int i)
        vector[double] get_death_rates_at_cell(int i)
        vector[double] get_all_x_coords()
        vector[double] get_all_death_rates()


cimport cython
from libcpp.vector cimport vector

cdef class PyGrid1D:

    cdef Grid1D* c_grid

    def __cinit__(self, double area_length_x, int cell_count_x,
                  double b, double d, double dd, int seed,
                  list initial_population, list death_values,
                  double death_cutoff_r, list birth_values,
                  bool periodic, double realtime_limit):

        cdef vector[double] cpp_initial_population = initial_population
        cdef vector[double] cpp_death_values = death_values
        cdef vector[double] cpp_birth_values = birth_values

        self.c_grid = new Grid1D(area_length_x, cell_count_x,
                                 b, d, dd, seed,
                                 cpp_initial_population,
                                 cpp_death_values,
                                 death_cutoff_r,
                                 cpp_birth_values,
                                 periodic, realtime_limit)

    def __dealloc__(self):
        if self.c_grid != NULL:  # Use NULL for C++ pointer comparison
            del self.c_grid

    def kill_random(self):
        self.c_grid.kill_random()

    def spawn_random(self):
        self.c_grid.spawn_random()

    def make_event(self):
        self.c_grid.make_event()

    def run_events(self, events: int):
        self.c_grid.run_events(events)

    def run_for(self, duration: float):
        self.c_grid.run_for(duration)

    def death_kernel(self, at: float) -> float:
        return self.c_grid.death_kernel(at)

    def birth_inverse_rcdf_kernel(self, at: float) -> float:
        return self.c_grid.birth_inverse_rcdf_kernel(at)

    def get_x_coords_at_cell(self, i: int) -> list:
        return list(self.c_grid.get_x_coords_at_cell(i))

    def get_death_rates_at_cell(self, i: int) -> list:
        return list(self.c_grid.get_death_rates_at_cell(i))

    def get_all_x_coords(self) -> list:
        return list(self.c_grid.get_all_x_coords())

    def get_all_death_rates(self) -> list:
        return list(self.c_grid.get_all_death_rates())

    @property
    def area_length_x(self):
        return self.c_grid.area_length_x

    @property
    def cell_count_x(self):
        return self.c_grid.cell_count_x

    @property
    def b(self):
        return self.c_grid.b

    @property
    def d(self):
        return self.c_grid.d

    @property
    def dd(self):
        return self.c_grid.dd

    @property
    def seed(self):
        return self.c_grid.seed

    @property
    def initial_population_x(self):
        return list(self.c_grid.initial_population_x)

    @property
    def death_y(self):
        return list(self.c_grid.death_y)

    @property
    def death_cutoff_r(self):
        return self.c_grid.death_cutoff_r

    @property
    def death_spline_nodes(self):
        return self.c_grid.death_spline_nodes

    @property
    def death_step(self):
        return self.c_grid.death_step

    @property
    def birth_inverse_rcdf_y(self):
        return list(self.c_grid.birth_inverse_rcdf_y)

    @property
    def birth_inverse_rcdf_nodes(self):
        return self.c_grid.birth_inverse_rcdf_nodes

    @property
    def birth_inverse_rcdf_step(self):
        return self.c_grid.birth_inverse_rcdf_step

    @property
    def periodic(self):
        return self.c_grid.periodic

    @property
    def realtime_limit(self):
        return self.c_grid.realtime_limit

    @property
    def realtime_limit_reached(self):
        return self.c_grid.realtime_limit_reached