#cython: language_level=3

# We may import some standard Cython / Python definitions
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
# If you need memory views, you can also: from cython cimport view

#######################################################
# 1) Declare std::array<T,N> manually for each needed
#    combination: T in {double,int}, N in {1,2,3}.
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
# 2) Provide helper functions to convert from Python
#    to these std::array types.  No cdef in loops!
#######################################################

cdef arrayDouble1 pyToStdArrayDouble1(list arr) except *:
    """
    Convert a Python list [x] to std::array<double,1>.
    """
    cdef arrayDouble1 result = arrayDouble1()
    result[0] = <double>arr[0]
    return result

cdef arrayDouble2 pyToStdArrayDouble2(list arr) except *:
    """
    Convert [x,y] to std::array<double,2>.
    """
    cdef arrayDouble2 result = arrayDouble2()
    result[0] = <double>arr[0]
    result[1] = <double>arr[1]
    return result

cdef arrayDouble3 pyToStdArrayDouble3(list arr) except *:
    """
    Convert [x,y,z] to std::array<double,3>.
    """
    cdef arrayDouble3 result = arrayDouble3()
    result[0] = <double>arr[0]
    result[1] = <double>arr[1]
    result[2] = <double>arr[2]
    return result

cdef arrayInt1 pyToStdArrayInt1(list arr) except *:
    """
    Convert [i] to std::array<int,1>.
    """
    cdef arrayInt1 result = arrayInt1()
    result[0] = <int>arr[0]
    return result

cdef arrayInt2 pyToStdArrayInt2(list arr) except *:
    """
    Convert [i,j] to std::array<int,2>.
    """
    cdef arrayInt2 result = arrayInt2()
    result[0] = <int>arr[0]
    result[1] = <int>arr[1]
    return result

cdef arrayInt3 pyToStdArrayInt3(list arr) except *:
    """
    Convert [i,j,k] to std::array<int,3>.
    """
    cdef arrayInt3 result = arrayInt3()
    result[0] = <int>arr[0]
    result[1] = <int>arr[1]
    result[2] = <int>arr[2]
    return result

#######################################################
# 3) Convert Python lists to std::vector<double> etc.
#######################################################

cdef vector[double] pyListToVectorDouble(object pyList) except *:
    """
    Convert a Python iterable of floats into a std::vector<double>.
    Ensures no cdef declarations happen inside the loop.
    """
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
    """
    Convert a list of lists of floats -> vector<vector<double>>.
    """
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
    """
    For e.g. death_x[s1][s2], stored as vector<vector<vector<double>>> in C++.
    """
    cdef int s1_count = len(pyList)
    cdef vector[ vector[ vector[double] ] ] out3
    out3.resize(s1_count)

    cdef object third_level
    cdef vector[double] tempVec

    cdef int s1, s2_count, s2, iSize
    cdef object second_level
    for s1 in range(s1_count):
        second_level = pyList[s1]
        s2_count = len(second_level)
        out3[s1].resize(s2_count)
        for s2 in range(s2_count):
            third_level = second_level[s2]
            tempVec = pyListToVectorDouble(third_level)
            out3[s1][s2] = tempVec
    return out3

#######################################################
# 4) Convert Python -> initialCoords for placeInitialPopulations
#    dimension-specific
#######################################################

cdef vector[ vector[ arrayDouble1 ] ] pyToInitialCoordsD1(object pyCoords) except *:
    """
    pyCoords[s] is a list of positions in 1D: [[x1], [x2], ...].
    We'll build vector[ vector[arrayDouble1] ].
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble1 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list singlePos
    cdef arrayDouble1 arr1
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            singlePos = posList[i]
            arr1 = pyToStdArrayDouble1(singlePos)
            result[s][i] = arr1
    return result

cdef vector[ vector[ arrayDouble2 ] ] pyToInitialCoordsD2(object pyCoords) except *:
    """
    pyCoords[s] = list of [ [x,y], [x2,y2], ... ] for species s.
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble2 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xy
    cdef arrayDouble2 arr2
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xy = posList[i]
            arr2 = pyToStdArrayDouble2(xy)
            result[s][i] = arr2
    return result

cdef vector[ vector[ arrayDouble3 ] ] pyToInitialCoordsD3(object pyCoords) except *:
    """
    pyCoords[s] = list of [ [x,y,z], [x2,y2,z2], ... ] for species s.
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[ vector[ arrayDouble3 ] ] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xyz
    cdef arrayDouble3 arr3
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xyz = posList[i]
            arr3 = pyToStdArrayDouble3(xyz)
            result[s][i] = arr3
    return result

#######################################################
# 5) Declare the extern C++ classes: Grid<1>, Grid<2>, Grid<3>
#######################################################

cdef extern from "SpatialBirthDeath.h":

    cdef cppclass Grid1 "Grid<1>":
        Grid1(int M_,
              arrayDouble1 areaLen,
              arrayInt1 cellCount_,
              bool isPeriodic,
              vector[double]  &birthRates,
              vector[double]  &deathRates,
              vector[double]  &ddMatrix,
              vector[ vector[double] ]  &birthX,
              vector[ vector[double] ]  &birthY,
              vector[ vector[ vector[double] ] ]  &deathX_,
              vector[ vector[ vector[double] ] ]  &deathY_,
              vector[double]  &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placeInitialPopulations(
            vector[ vector[ arrayDouble1 ] ]  &initialCoords
        ) except +

        void computeInitialDeathRates() except +
        void make_event() except +
        void spawn_random() except +
        void kill_random() except +

        double total_birth_rate
        double total_death_rate
        int    total_population
        double time
        int    event_count

    cdef cppclass Grid2 "Grid<2>":
        Grid2(int M_,
              arrayDouble2 areaLen,
              arrayInt2 cellCount_,
              bool isPeriodic,
              vector[double]  &birthRates,
              vector[double]  &deathRates,
              vector[double]  &ddMatrix,
              vector[ vector[double] ]  &birthX,
              vector[ vector[double] ]  &birthY,
              vector[ vector[ vector[double] ] ]  &deathX_,
              vector[ vector[ vector[double] ] ]  &deathY_,
              vector[double]  &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placeInitialPopulations(
            vector[ vector[ arrayDouble2 ] ]  &initialCoords
        ) except +

        void computeInitialDeathRates() except +
        void make_event() except +
        void spawn_random() except +
        void kill_random() except +

        double total_birth_rate
        double total_death_rate
        int    total_population
        double time
        int    event_count

    cdef cppclass Grid3 "Grid<3>":
        Grid3(int M_,
              arrayDouble3 areaLen,
              arrayInt3 cellCount_,
              bool isPeriodic,
              vector[double]  &birthRates,
              vector[double]  &deathRates,
              vector[double]  &ddMatrix,
              vector[ vector[double] ]  &birthX,
              vector[ vector[double] ]  &birthY,
              vector[ vector[ vector[double] ] ]  &deathX_,
              vector[ vector[ vector[double] ] ]  &deathY_,
              vector[double]  &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placeInitialPopulations(
            vector[ vector[ arrayDouble3 ] ] &initialCoords
        ) except +

        void computeInitialDeathRates() except +
        void make_event() except +
        void spawn_random() except +
        void kill_random() except +

        double total_birth_rate
        double total_death_rate
        int    total_population
        double time
        int    event_count


#######################################################
# 6) Python-visible wrapper classes for each dimension
#######################################################

cdef class PyGrid1:
    """
    A Python wrapper around the C++ Grid<1> class.
    """
    cdef Grid1* cpp_grid  # Owned pointer

    def __cinit__(self,
                  M,
                  areaLen,         # [x_size]
                  cellCount,       # [nx]
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
        # Convert python objects to the C++ types:
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

    def placeInitialPopulations(self, initCoords):
        """
        initCoords[s] = list of [ [x], [x2], ... ] for species s.
        """
        cdef vector[ vector[ arrayDouble1 ] ] c_init = pyToInitialCoordsD1(initCoords)
        self.cpp_grid.placeInitialPopulations(c_init)

    def computeInitialDeathRates(self):
        self.cpp_grid.computeInitialDeathRates()

    def make_event(self):
        self.cpp_grid.make_event()

    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

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


cdef class PyGrid2:
    """
    A Python wrapper around the C++ Grid<2> class.
    """
    cdef Grid2* cpp_grid

    def __cinit__(self,
                  M,
                  areaLen,         # [x_size, y_size]
                  cellCount,       # [nx, ny]
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

    def placeInitialPopulations(self, initCoords):
        """
        initCoords[s] = list of [ [x,y], [x2,y2], ... ] for species s.
        """
        cdef vector[ vector[ arrayDouble2 ] ] c_init = pyToInitialCoordsD2(initCoords)
        self.cpp_grid.placeInitialPopulations(c_init)

    def computeInitialDeathRates(self):
        self.cpp_grid.computeInitialDeathRates()

    def make_event(self):
        self.cpp_grid.make_event()

    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

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


cdef class PyGrid3:
    """
    A Python wrapper around the C++ Grid<3> class.
    """
    cdef Grid3* cpp_grid

    def __cinit__(self,
                  M,
                  areaLen,         # [x_size, y_size, z_size]
                  cellCount,       # [nx, ny, nz]
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

    def placeInitialPopulations(self, initCoords):
        """
        initCoords[s] = list of [ [x,y,z], [..], ... ] for species s.
        """
        cdef vector[ vector[ arrayDouble3 ] ] c_init = pyToInitialCoordsD3(initCoords)
        self.cpp_grid.placeInitialPopulations(c_init)

    def computeInitialDeathRates(self):
        self.cpp_grid.computeInitialDeathRates()

    def make_event(self):
        self.cpp_grid.make_event()

    def spawn_random(self):
        self.cpp_grid.spawn_random()

    def kill_random(self):
        self.cpp_grid.kill_random()

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
