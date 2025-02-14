#cython: language_level=3

# We may import some standard Cython / Python definitions
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free

#######################################################
# 1) Declare std::array<T,N> for needed combos
#######################################################
cdef extern from "<array>" namespace "std" nogil:

    # -- double, dimension=1 --
    cdef cppclass arrayDouble1 "std::array<double, 1>":
        arrayDouble1() except +
        double& operator[](size_t) except +

    # -- double, dimension=2 --
    cdef cppclass arrayDouble2 "std::array<double, 2>":
        arrayDouble2() except +
        double& operator[](size_t) except +

    # -- double, dimension=3 --
    cdef cppclass arrayDouble3 "std::array<double, 3>":
        arrayDouble3() except +
        double& operator[](size_t) except +

    # -- int, dimension=1 --
    cdef cppclass arrayInt1 "std::array<int, 1>":
        arrayInt1() except +
        int& operator[](size_t) except +

    # -- int, dimension=2 --
    cdef cppclass arrayInt2 "std::array<int, 2>":
        arrayInt2() except +
        int& operator[](size_t) except +

    # -- int, dimension=3 --
    cdef cppclass arrayInt3 "std::array<int, 3>":
        arrayInt3() except +
        int& operator[](size_t) except +

#######################################################
# 2) Provide helper functions to convert Python -> std::array
#######################################################

cdef arrayDouble1 pyToStdArrayDouble1(list arr) except *:
    cdef arrayDouble1 result = arrayDouble1()
    result[0] = <double>arr[0]
    return result

cdef arrayDouble2 pyToStdArrayDouble2(list arr) except *:
    cdef arrayDouble2 result = arrayDouble2()
    result[0] = <double>arr[0]
    result[1] = <double>arr[1]
    return result

cdef arrayDouble3 pyToStdArrayDouble3(list arr) except *:
    cdef arrayDouble3 result = arrayDouble3()
    result[0] = <double>arr[0]
    result[1] = <double>arr[1]
    result[2] = <double>arr[2]
    return result

cdef arrayInt1 pyToStdArrayInt1(list arr) except *:
    cdef arrayInt1 result = arrayInt1()
    result[0] = <int>arr[0]
    return result

cdef arrayInt2 pyToStdArrayInt2(list arr) except *:
    cdef arrayInt2 result = arrayInt2()
    result[0] = <int>arr[0]
    result[1] = <int>arr[1]
    return result

cdef arrayInt3 pyToStdArrayInt3(list arr) except *:
    cdef arrayInt3 result = arrayInt3()
    result[0] = <int>arr[0]
    result[1] = <int>arr[1]
    result[2] = <int>arr[2]
    return result

#######################################################
# 3) Convert Python lists to std::vector<double> etc.
#######################################################

cdef vector[double] pyListToVectorDouble(object pyList) except *:
    cdef int n = len(pyList)
    cdef vector[double] vec
    vec.resize(n)
    cdef int i
    cdef double val
    for i in range(n):
        val = pyList[i]
        vec[i] = val
    return vec

cdef vector[ vector[double] ] pyListOfListToVectorVectorDouble(object pyList) except *:
    cdef int outer_size = len(pyList)
    cdef vector[ vector[double] ] result
    result.resize(outer_size)

    cdef int i, inner_size
    cdef object inner_list
    cdef int j
    cdef double val
    for i in range(outer_size):
        inner_list = pyList[i]
        inner_size = len(inner_list)
        result[i].resize(inner_size)
        for j in range(inner_size):
            val = inner_list[j]
            result[i][j] = val
    return result

cdef vector[ vector[ vector[double] ] ] pyListOfListOfListToVector3Double(object pyList) except *:
    cdef int s1_count = len(pyList)
    cdef vector[ vector[ vector[double] ] ] out3
    out3.resize(s1_count)

    cdef object second_level
    cdef object third_level
    cdef int s1, s2_count, s2
    for s1 in range(s1_count):
        second_level = pyList[s1]
        s2_count = len(second_level)
        out3[s1].resize(s2_count)
        for s2 in range(s2_count):
            third_level = second_level[s2]
            out3[s1][s2] = pyListToVectorDouble(third_level)
    return out3

#######################################################
# 4) Convert Python -> data for placePopulation(...)
#    dimension-specific
#######################################################
# We'll keep these for convenience if the user wants to place multiple coords.

cdef vector[ vector[ arrayDouble1 ] ] pyToCoordsD1(object pyCoords) except *:
    """
    pyCoords[s] = list of positions in 1D, e.g. [[x1], [x2], ...].
    We'll build vector[ vector[arrayDouble1] ].
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble1 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list singlePos
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            singlePos = posList[i]
            result[s][i] = pyToStdArrayDouble1(singlePos)
    return result

cdef vector[ vector[ arrayDouble2 ] ] pyToCoordsD2(object pyCoords) except *:
    """
    pyCoords[s] = list of [ [x,y], [x2,y2], ... ] for species s.
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble2 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xy
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xy = posList[i]
            result[s][i] = pyToStdArrayDouble2(xy)
    return result

cdef vector[ vector[ arrayDouble3 ] ] pyToCoordsD3(object pyCoords) except *:
    """
    pyCoords[s] = list of [ [x,y,z], [..], ... ] for species s.
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble3 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xyz
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xyz = posList[i]
            result[s][i] = pyToStdArrayDouble3(xyz)
    return result

#######################################################
# 5) Expose Cell<DIM> (for reading only)
#######################################################
cdef extern from "SpatialBirthDeath.h":
    cdef cppclass Cell1 "Cell<1>":
        vector[ vector[arrayDouble1] ] coords
        vector[ vector[double] ]       deathRates
        vector[int]                    population
        vector[double]                cellBirthRateBySpecies
        vector[double]                cellDeathRateBySpecies
        double                        cellBirthRate
        double                        cellDeathRate

cdef extern from "SpatialBirthDeath.h":
    cdef cppclass Cell2 "Cell<2>":
        vector[ vector[arrayDouble2] ] coords
        vector[ vector[double] ]       deathRates
        vector[int]                    population
        vector[double]                cellBirthRateBySpecies
        vector[double]                cellDeathRateBySpecies
        double                        cellBirthRate
        double                        cellDeathRate

cdef extern from "SpatialBirthDeath.h":
    cdef cppclass Cell3 "Cell<3>":
        vector[ vector[arrayDouble3] ] coords
        vector[ vector[double] ]       deathRates
        vector[int]                    population
        vector[double]                cellBirthRateBySpecies
        vector[double]                cellDeathRateBySpecies
        double                        cellBirthRate
        double                        cellDeathRate

#######################################################
# 6) Expose Grid<DIM> classes with new methods
#######################################################
cdef extern from "SpatialBirthDeath.h":

    cdef cppclass Grid1 "Grid<1>":
        Grid1(int M_,
              arrayDouble1 areaLen,
              arrayInt1 cellCount_,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[ vector[double] ] &birthX,
              vector[ vector[double] ] &birthY,
              vector[ vector[ vector[double] ] ] &deathX_,
              vector[ vector[ vector[double] ] ] &deathY_,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        # New approach: placePopulation, spawn_at, kill_at
        void placePopulation(const vector[ vector[ arrayDouble1 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble1 &inPos) except +
        void kill_at(int s, const arrayInt1 &cIdx, int victimIdx) except +

        # The random event methods
        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble1]] get_all_particle_coords() except +

        # Exposed fields:
        double total_birth_rate
        double total_death_rate
        int    total_num_cells
        int    total_population
        double time
        int    event_count
        vector[Cell1] cells

    cdef cppclass Grid2 "Grid<2>":
        Grid2(int M_,
              arrayDouble2 areaLen,
              arrayInt2 cellCount_,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[ vector[double] ] &birthX,
              vector[ vector[double] ] &birthY,
              vector[ vector[ vector[double] ] ] &deathX_,
              vector[ vector[ vector[double] ] ] &deathY_,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placePopulation(const vector[ vector[ arrayDouble2 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble2 &inPos) except +
        void kill_at(int s, const arrayInt2 &cIdx, int victimIdx) except +

        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble2]] get_all_particle_coords() except +

        double total_birth_rate
        double total_death_rate
        int    total_num_cells
        int    total_population
        double time
        int    event_count
        vector[Cell2] cells

    cdef cppclass Grid3 "Grid<3>":
        Grid3(int M_,
              arrayDouble3 areaLen,
              arrayInt3 cellCount_,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[ vector[double] ] &birthX,
              vector[ vector[double] ] &birthY,
              vector[ vector[ vector[double] ] ] &deathX_,
              vector[ vector[ vector[double] ] ] &deathY_,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placePopulation(const vector[ vector[ arrayDouble3 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble3 &inPos) except +
        void kill_at(int s, const arrayInt3 &cIdx, int victimIdx) except +

        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble3]] get_all_particle_coords() except +

        double total_birth_rate
        double total_death_rate
        int    total_num_cells
        int    total_population
        double time
        int    event_count
        vector[Cell3] cells

#######################################################
# 7) Python wrapper classes
#######################################################

#==================== Grid<1> ====================#
cdef class PyGrid1:
    cdef Grid1* cpp_grid  # Owned pointer

    def __cinit__(self,
                  M,
                  areaLen,    # e.g. [25.0]
                  cellCount,  # e.g. [25]
                  isPeriodic,
                  birthRates,
                  deathRates,
                  ddMatrix,
                  birthX,
                  birthY,
                  deathX_,
                  deathY_,
                  cutoffs,
                  seed,
                  rtimeLimit):
        cdef arrayDouble1 c_areaLen = pyToStdArrayDouble1(areaLen)
        cdef arrayInt1    c_cellCount = pyToStdArrayInt1(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX_)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY_)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid1(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    # ------ New function: placePopulation ------
    def placePopulation(self, initCoords):
        """
        initCoords[s] = list of [ [x1], [x2], ... ] for species s.
        """
        cdef vector[ vector[arrayDouble1] ] c_init = pyToCoordsD1(initCoords)
        self.cpp_grid.placePopulation(c_init)

    # ------ spawn_at / kill_at ------
    def spawn_at(self, s, pos):
        """
        s: int (species index)
        pos: [x]
        """
        cdef arrayDouble1 cpos = pyToStdArrayDouble1(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        """
        s: int (species index)
        cell_idx: [i]
        pos: [x]
        """
        cdef arrayInt1 cc = pyToStdArrayInt1(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # ------ random events ------
    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

    def make_event(self):
        self.cpp_grid.make_event()

    def run_events(self, n):
        self.cpp_grid.run_events(n)

    def run_for(self, duration):
        self.cpp_grid.run_for(duration)

    # ------ read-only properties ------
    @property
    def total_birth_rate(self):
        return self.cpp_grid.total_birth_rate

    @property
    def total_death_rate(self):
        return self.cpp_grid.total_death_rate

    @property
    def total_population(self):
        return self.cpp_grid.total_population

    @property
    def time(self):
        return self.cpp_grid.time

    @property
    def event_count(self):
        return self.cpp_grid.event_count

    def get_num_cells(self):
        return self.cpp_grid.total_num_cells

    # ------ Access cell data ------
    def get_cell_coords(self, cell_index, species_idx):
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[arrayDouble1] coords_vec = cptr.coords[species_idx]
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append(coords_vec[i][0])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[double] drates = cptr.deathRates[species_idx]
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_cell_population(self, cell_index):
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[int] pop = cptr.population
        cdef int m = pop.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = pop[s]
        return out

    def get_cell_birth_rate(self, cell_index):
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellBirthRate

    def get_cell_death_rate(self, cell_index):
        cdef Cell1 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellDeathRate

    def get_all_particle_coords(self):
        """
        Returns aggregated coordinates for all particles in the grid,
        grouped by species. For a 1D grid, each coordinate is a float.

        Returns:
            A list of lists: one list per species, where each inner list
            contains the x-coordinate (a float) for every particle.
        """
        cdef vector[vector[arrayDouble1]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef int ns = c_all.size()
        py_out = []
        cdef int i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                # For 1D, each coordinate is stored in a std::array<double,1>
                species_coords.append(c_all[i][j][0])
            py_out.append(species_coords)
        return py_out


#==================== Grid<2> ====================#
cdef class PyGrid2:
    cdef Grid2* cpp_grid

    def __cinit__(self,
                  M,
                  areaLen,    # [width, height]
                  cellCount,  # [nx, ny]
                  isPeriodic,
                  birthRates,
                  deathRates,
                  ddMatrix,
                  birthX,
                  birthY,
                  deathX_,
                  deathY_,
                  cutoffs,
                  seed,
                  rtimeLimit):
        cdef arrayDouble2 c_areaLen = pyToStdArrayDouble2(areaLen)
        cdef arrayInt2    c_cellCount = pyToStdArrayInt2(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX_)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY_)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid2(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    # ------ placePopulation ------
    def placePopulation(self, initCoords):
        """
        initCoords[s] = list of [ [x,y], [..], ... ] for species s
        """
        cdef vector[ vector[arrayDouble2] ] c_init = pyToCoordsD2(initCoords)
        self.cpp_grid.placePopulation(c_init)

    # ------ spawn_at / kill_at ------
    def spawn_at(self, s, pos):
        """
        s: int
        pos: [x, y]
        """
        cdef arrayDouble2 cpos = pyToStdArrayDouble2(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        """
        s: int
        cell_idx: [ix, iy]
        pos: [x, y]
        """
        cdef arrayInt2 cc = pyToStdArrayInt2(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # ------ random events ------
    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

    def make_event(self):
        self.cpp_grid.make_event()

    def run_events(self, n):
        self.cpp_grid.run_events(n)

    def run_for(self, duration):
        self.cpp_grid.run_for(duration)

    # ------ read-only properties ------
    @property
    def total_birth_rate(self):
        return self.cpp_grid.total_birth_rate

    @property
    def total_death_rate(self):
        return self.cpp_grid.total_death_rate

    @property
    def total_population(self):
        return self.cpp_grid.total_population

    @property
    def time(self):
        return self.cpp_grid.time

    @property
    def event_count(self):
        return self.cpp_grid.event_count

    def get_num_cells(self):
        return self.cpp_grid.total_num_cells

    # ------ Access cell data ------
    def get_cell_coords(self, cell_index, species_idx):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[arrayDouble2] coords_vec = cptr.coords[species_idx]
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append([coords_vec[i][0], coords_vec[i][1]])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[double] drates = cptr.deathRates[species_idx]
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_cell_population(self, cell_index):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[int] pop = cptr.population
        cdef int m = pop.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = pop[s]
        return out

    def get_cell_birth_rate(self, cell_index):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellBirthRate

    def get_cell_death_rate(self, cell_index):
        cdef Cell2 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellDeathRate

    def get_all_particle_coords(self):
        """
        Returns aggregated coordinates for all particles in the grid,
        grouped by species. For a 2D grid, each coordinate is a list [x, y].

        Returns:
            A list of lists: one list per species, where each inner list
            contains the [x, y] coordinates of a particle.
        """
        cdef vector[vector[arrayDouble2]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef int ns = c_all.size()
        py_out = []
        cdef int i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                species_coords.append([c_all[i][j][0], c_all[i][j][1]])
            py_out.append(species_coords)
        return py_out

#==================== Grid<3> ====================#
cdef class PyGrid3:
    cdef Grid3* cpp_grid

    def __cinit__(self,
                  M,
                  areaLen,     # [x_size, y_size, z_size]
                  cellCount,   # [nx, ny, nz]
                  isPeriodic,
                  birthRates,
                  deathRates,
                  ddMatrix,
                  birthX,
                  birthY,
                  deathX_,
                  deathY_,
                  cutoffs,
                  seed,
                  rtimeLimit):
        cdef arrayDouble3 c_areaLen = pyToStdArrayDouble3(areaLen)
        cdef arrayInt3    c_cellCount = pyToStdArrayInt3(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX_)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY_)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid3(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    # ------ placePopulation ------
    def placePopulation(self, initCoords):
        """
        initCoords[s] = list of [ [x,y,z], [..], ... ] for species s.
        """
        cdef vector[ vector[arrayDouble3] ] c_init = pyToCoordsD3(initCoords)
        self.cpp_grid.placePopulation(c_init)

    # ------ spawn_at / kill_at ------
    def spawn_at(self, s, pos):
        """
        s: int
        pos: [x, y, z]
        """
        cdef arrayDouble3 cpos = pyToStdArrayDouble3(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        """
        s: int
        cell_idx: [ix, iy, iz]
        pos: [x, y, z]
        """
        cdef arrayInt3 cc = pyToStdArrayInt3(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # ------ random events ------
    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

    def make_event(self):
        self.cpp_grid.make_event()

    def run_events(self, n):
        self.cpp_grid.run_events(n)

    def run_for(self, duration):
        self.cpp_grid.run_for(duration)

    # ------ read-only properties ------
    @property
    def total_birth_rate(self):
        return self.cpp_grid.total_birth_rate

    @property
    def total_death_rate(self):
        return self.cpp_grid.total_death_rate

    @property
    def total_population(self):
        return self.cpp_grid.total_population

    @property
    def time(self):
        return self.cpp_grid.time

    @property
    def event_count(self):
        return self.cpp_grid.event_count

    def get_num_cells(self):
        return self.cpp_grid.total_num_cells

    # ------ Access cell data ------
    def get_cell_coords(self, cell_index, species_idx):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[arrayDouble3] coords_vec = cptr.coords[species_idx]
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append([coords_vec[i][0],
                        coords_vec[i][1],
                        coords_vec[i][2]])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[double] drates = cptr.deathRates[species_idx]
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_cell_population(self, cell_index):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        cdef vector[int] pop = cptr.population
        cdef int m = pop.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = pop[s]
        return out

    def get_cell_birth_rate(self, cell_index):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellBirthRate

    def get_cell_death_rate(self, cell_index):
        cdef Cell3 * cptr = &self.cpp_grid.cells[cell_index]
        return cptr.cellDeathRate

    def get_all_particle_coords(self):
        """
        Returns aggregated coordinates for all particles in the grid,
        grouped by species. For a 3D grid, each coordinate is a list [x, y, z].

        Returns:
            A list of lists: one list per species, where each inner list
            contains the [x, y, z] coordinates of a particle.
        """
        cdef vector[vector[arrayDouble3]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef int ns = c_all.size()
        py_out = []
        cdef int i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                species_coords.append([c_all[i][j][0],
                                       c_all[i][j][1],
                                       c_all[i][j][2]])
            py_out.append(species_coords)
        return py_out
