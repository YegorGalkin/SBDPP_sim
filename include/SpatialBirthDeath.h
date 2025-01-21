#ifndef SPATIAL_BIRTH_DEATH_H
#define SPATIAL_BIRTH_DEATH_H

/**
 * \file SpatialBirthDeath.h
 * \brief Header file for a spatial birth-death point process simulator.
 *
 * This simulator models spatially explicit discrete population dynamics
 * in a region of dimension DIM. It supports one or more species with
 * birth-death events, user-defined dispersal and competition kernels,
 * and can optionally handle periodic boundary conditions.
 *
 * \date 2025-01-20
 */

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <stdexcept>



/**
 * \brief Performs a linear interpolation on tabulated (x,y) data.
 *
 * Given a set of x-values \p xVals and their corresponding y-values \p yVals,
 * and an input \p x, this function returns the linearly interpolated value.
 * If \p x is outside the range of xVals, the boundary y-value is returned.
 *
 * \param xVals Sorted array of x-values.
 * \param yVals Corresponding array of y-values.
 * \param x     The input at which to interpolate.
 * \return      The interpolated y-value.
 */
inline double linearInterpolate(const std::vector<double> &xVals,
                                const std::vector<double> &yVals,
                                double x)
{
    // If below or above the range, return boundary values
    if (x <= xVals.front()) return yVals.front();
    if (x >= xVals.back())  return yVals.back();

    // Find the position of x in xVals
    auto it = std::lower_bound(xVals.begin(), xVals.end(), x);
    size_t idx = (it - xVals.begin());

    // Edge case: x is exactly xVals[0]
    if (idx == 0) return yVals[0];

    // Take neighboring points x1, x2 and linearly interpolate
    double x1 = xVals[idx - 1];
    double x2 = xVals[idx];
    double y1 = yVals[idx - 1];
    double y2 = yVals[idx];
    double t = (x - x1)/(x2 - x1);
    return y1 + t*(y2 - y1);
}

/**
 * \brief A helper function to recursively iterate over neighboring
 *        cells in DIM dimensions.
 *
 * This function, \p forNeighborsRecur, is used internally by \p forNeighbors
 * to build multi-indices around a "center" index. It calls \p func for each
 * possible neighbor index within the \p cull range in each dimension.
 *
 * \tparam DIM  The dimension of the simulation space.
 * \tparam FUNC A callable type that accepts a single parameter of type
 *              `const std::array<int, DIM> &`.
 *
 * \param center   The cell index around which we are collecting neighbors.
 * \param cull     The maximum offset in each dimension for neighbor iteration.
 * \param temp     A temporary array that accumulates the offsets.
 * \param dimIndex Current dimension index being processed.
 * \param func     The function to be called for each neighbor index.
 */
template<int DIM, typename FUNC>
void forNeighborsRecur(const std::array<int, DIM> &center,
                       const std::array<int, DIM> &cull,
                       std::array<int, DIM> &temp,
                       int dimIndex,
                       const FUNC &func)
{
    if (dimIndex == DIM) {
        // We have a complete multi-index in temp
        func(temp);
    }
    else {
        for (int offset = -cull[dimIndex]; offset <= cull[dimIndex]; ++offset) {
            temp[dimIndex] = center[dimIndex] + offset;
            forNeighborsRecur<DIM>(center, cull, temp, dimIndex+1, func);
        }
    }
}

/**
 * \brief Iterates over all neighboring cell indices within the specified
 *        cull range and applies a provided functor.
 *
 * Example usage:
 * \code
 *   std::array<int,2> center = {10, 20};
 *   std::array<int,2> cull   = {1, 1};
 *   forNeighbors<2>(center, cull, [&](const std::array<int,2> &nbr){
 *       // do something with nbr
 *   });
 * \endcode
 *
 * \tparam DIM  The dimension of the simulation space.
 * \tparam FUNC A callable type that accepts a single parameter of type
 *              `const std::array<int, DIM> &`.
 *
 * \param center The cell index around which we are collecting neighbors.
 * \param cull   The maximum offset in each dimension for neighbor iteration.
 * \param func   The function to be called for each neighbor index.
 */
template<int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &center,
                  const std::array<int, DIM> &cull,
                  const FUNC &func)
{
    std::array<int, DIM> temp;
    forNeighborsRecur<DIM>(center, cull, temp, 0, func);
}

/**
 * \brief Computes Euclidean distance between two points in DIM dimensions,
 *        with an option for periodic boundaries.
 *
 * For periodic boundaries, if the difference in any dimension is more than
 * half the domain length in that dimension, it is "wrapped" accordingly.
 *
 * \tparam DIM The dimension of the simulation space.
 *
 * \param a        The first point (array of length DIM).
 * \param b        The second point (array of length DIM).
 * \param length   The size of the domain in each dimension.
 * \param periodic If `true`, apply periodic boundary logic.
 * \return         The Euclidean distance between \p a and \p b.
 */
template<int DIM>
double distancePeriodic(const std::array<double,DIM> &a,
                        const std::array<double,DIM> &b,
                        const std::array<double,DIM> &length,
                        bool periodic)
{
    double dist2 = 0.0;
    for (int d = 0; d < DIM; d++) {
        double diff = a[d] - b[d];
        if (periodic) {
            double L = length[d];
            if      (diff >  0.5 * L) diff -= L;
            else if (diff < -0.5 * L) diff += L;
        }
        dist2 += diff * diff;
    }
    return std::sqrt(dist2);
}

/**
 * \brief The Cell data structure stores coordinates, death rates, and
 *        cached birth/death rates for each species in a single cell.
 *
 * For each species `s`:
 *   - \c coords[s][i] : coordinate of the i-th particle of species s
 *   - \c deathRates[s][i] : individual's death rate
 *   - \c population[s] : total number of individuals of species s in this cell
 *
 * Additionally, we store partial sums to speed up event selection:
 *   - \c cellBirthRateBySpecies[s]
 *   - \c cellDeathRateBySpecies[s]
 *   - \c cellBirthRate (sum of \c cellBirthRateBySpecies)
 *   - \c cellDeathRate (sum of \c cellDeathRateBySpecies)
 *
 * \tparam DIM The dimension of the simulation space.
 */
template<int DIM>
struct Cell
{
    /// \brief Particle positions per species: coords[s].
    std::vector< std::vector< std::array<double, DIM> > > coords;

    /// \brief Death rates for each particle of each species: deathRates[s].
    std::vector< std::vector<double> >                    deathRates;

    /// \brief Population count for each species in this cell: population[s].
    std::vector<int>                                      population;

    /// \brief Cached birth rates per species in this cell.
    std::vector<double> cellBirthRateBySpecies;

    /// \brief Cached death rates per species in this cell.
    std::vector<double> cellDeathRateBySpecies;

    /// \brief Summed birth rate across all species in this cell.
    double cellBirthRate;

    /// \brief Summed death rate across all species in this cell.
    double cellDeathRate;

    /**
     * \brief Constructor. Initializes rates to zero.
     */
    Cell()
        : cellBirthRate(0.0), cellDeathRate(0.0)
    {}

    /**
     * \brief Allocate internal storage for M species.
     * \param M The number of species.
     */
    void initSpecies(int M) {
        coords.resize(M);
        deathRates.resize(M);
        population.resize(M, 0);
        cellBirthRateBySpecies.resize(M, 0.0);
        cellDeathRateBySpecies.resize(M, 0.0);
    }
};

/**
 * \brief The Grid class organizes the entire simulation domain into a grid
 *        of cells. It holds parameters and data for birth and death rates,
 *        dispersal kernels, competition kernels, etc.
 *
 * ### Major Components:
 * - **Dimension**: `DIM` (templated)
 * - **Number of species**: `M`
 * - **Domain size**: `area_length[d]` for each dimension `d`
 * - **Cell count**: `cell_count[d]` for each dimension `d`
 * - **Periodicity**: If `true`, boundaries wrap around
 * - **Kernels**:
 *    - `birth_x[s], birth_y[s]`: radial birth/dispersal kernel (ICDF)
 *    - `death_x[s1][s2], death_y[s1][s2]`: radial competition kernel
 * - **Rates**:
 *    - `b[s]`: Poisson birth rate of species `s`
 *    - `d[s]`: Baseline death rate of species `s`
 *    - `dd[s1][s2]`: Interspecies competition magnitude
 * - **Cutoff**: maximum distance at which competition is considered
 * - **cull**: the cell-based truncation range in each dimension
 * - **Cells**: a `std::vector<Cell<DIM>>` of size `total_num_cells`
 * - **RNG**: a `std::mt19937` for reproducible random events
 * - **Time**: simulation time
 * - **Real-time limit**: a wall-clock cutoff to avoid hanging
 *
 * ### Event Simulation:
 * - We keep track of `total_birth_rate` and `total_death_rate`.
 * - On each event:
 *   1) We pick event type (birth or death) by relative ratio of birth vs. death.
 *   2a) If **birth**:
 *       - Discrete sample a cell proportional to its `cellBirthRate`.
 *       - Within that cell, pick a species proportional to `cellBirthRateBySpecies`.
 *       - Pick a random parent in that species (uniform among individuals).
 *       - Sample a distance from the birth kernel and pick a random direction.
 *       - Attempt to place the offspring, respecting periodic or open boundary.
 *   2b) If **death**:
 *       - Discrete sample a cell proportional to its `cellDeathRate`.
 *       - Pick a species in that cell proportional to `cellDeathRateBySpecies`.
 *       - Pick an individual in that species proportional to its `deathRate`.
 *       - Remove that individual and update neighbor interactions.
 *
 * \tparam DIM The dimension of the simulation space (1, 2, or 3).
 */
template<int DIM>
class Grid
{
public:
    /// \brief Number of species
    int M;

    /// \brief Size of the simulation area along each dimension
    std::array<double, DIM> area_length;

    /// \brief Number of cells along each dimension
    std::array<int,    DIM> cell_count;

    /// \brief Flag indicating whether the domain is periodic
    bool periodic;

    // -------------------------------------------------------
    //  Per-species baseline rates
    // -------------------------------------------------------
    /// \brief Birth rate b[s] for each species
    std::vector<double> b;

    /// \brief Baseline death rate d[s] for each species
    std::vector<double> d;

    // -------------------------------------------------------
    //  Pairwise interaction coefficients
    // -------------------------------------------------------
    /// \brief dd[s1][s2]: Magnitude of competition of s2 on s1
    std::vector< std::vector<double> > dd;

    // -------------------------------------------------------
    //  Kernels for birth & death (dispersal / competition)
    // -------------------------------------------------------
    /// \brief birth_x[s]: x-values (CDF domain) for species s
    std::vector< std::vector<double> > birth_x;

    /// \brief birth_y[s]: y-values (CDF range) for species s
    std::vector< std::vector<double> > birth_y;

    /// \brief death_x[s1][s2]: x-values for death kernel of (s1,s2)
    std::vector< std::vector< std::vector<double> > > death_x;

    /// \brief death_y[s1][s2]: y-values for death kernel of (s1,s2)
    std::vector< std::vector< std::vector<double> > > death_y;

    // -------------------------------------------------------
    //  Cell-based cutoff and cull
    // -------------------------------------------------------
    /// \brief cutoff[s1][s2]: distance beyond which interactions are ignored
    std::vector< std::vector<double> > cutoff;

    /**
     * \brief cull[s1][s2][dim]: # of cells in +/- each dimension
     *        we need to iterate for neighbor checks.
     */
    std::vector< std::vector< std::array<int,DIM> > > cull;

    // -------------------------------------------------------
    //  Cells
    // -------------------------------------------------------
    /// \brief Array of all cells in the grid
    std::vector< Cell<DIM> > cells;

    /// \brief Total number of cells = product of cell_count[dim]
    int total_num_cells;

    // -------------------------------------------------------
    //  Summaries for discrete sampling
    // -------------------------------------------------------
    /// \brief sum over all cells of cellBirthRate
    double total_birth_rate;

    /// \brief sum over all cells of cellDeathRate
    double total_death_rate;

    /// \brief total population of all species
    int    total_population;

    // -------------------------------------------------------
    //  Random number generator, time, and real-time limit
    // -------------------------------------------------------
    std::mt19937 rng;    ///< Mersenne Twister random generator
    double time;         ///< simulation time
    int    event_count;  ///< total number of events executed

    std::chrono::system_clock::time_point init_time; ///< to track wall-clock
    double realtime_limit;                           ///< maximum allowed wall-clock (seconds)
    bool   realtime_limit_reached;                   ///< set to true if we exceed realtime_limit

public:
    /**
     * \brief Main constructor for the simulation grid.
     *
     * \param M_         Number of species
     * \param areaLen    Physical length of each dimension
     * \param cellCount_ Number of cells in each dimension
     * \param isPeriodic Whether domain is periodic
     * \param birthRates Per-species birth rate b[s]
     * \param deathRates Per-species baseline death rate d[s]
     * \param ddMatrix   Flattened MxM matrix of dd[s1][s2]
     * \param birthX     birth_x[s] - table of x-values for species s
     * \param birthY     birth_y[s] - table of y-values for species s
     * \param deathX_    death_x[s1][s2] - x-values for each pair
     * \param deathY_    death_y[s1][s2] - y-values for each pair
     * \param cutoffs    Flattened MxM matrix of cutoff distances
     * \param seed       RNG seed
     * \param rtimeLimit Real-time limit in seconds
     */
    Grid(int M_,
         const std::array<double,DIM> &areaLen,
         const std::array<int,DIM> &cellCount_,
         bool isPeriodic,
         const std::vector<double> &birthRates,
         const std::vector<double> &deathRates,
         const std::vector<double> &ddMatrix,
         const std::vector< std::vector<double> > &birthX,
         const std::vector< std::vector<double> > &birthY,
         const std::vector< std::vector< std::vector<double> > > &deathX_,
         const std::vector< std::vector< std::vector<double> > > &deathY_,
         const std::vector<double> &cutoffs,
         int seed,
         double rtimeLimit
    );

    /**
     * \brief Flatten a multi-dimensional cell index to a 1D index.
     * \param idx An array of length DIM, specifying the cell coordinate.
     * \return The flattened integer index into the `cells` array.
     */
    int flattenIdx(const std::array<int,DIM> &idx) const;

    /**
     * \brief Wraps cell index `i` in dimension \p dim if the domain is periodic.
     * \param i   The index in that dimension (could be negative or out of range).
     * \param dim Which dimension we are working with.
     * \return    The wrapped (or clamped) index.
     */
    int wrapIndex(int i, int dim) const;

    bool inDomain(const std::array<int,DIM>& idx) const;

    /**
     * \brief Return a reference to the cell at raw index \p raw, applying wrap
     *        if periodic is enabled.
     * \param raw A possibly out-of-range index array [i0, i1, ..., i_{DIM-1}].
     * \return     A reference to the corresponding `Cell<DIM>`.
     */
    Cell<DIM>& cellAt(const std::array<int,DIM> &raw);

    /**
     * \brief Evaluate the birth (dispersal) kernel of species s at x.
     * \param s The species index.
     * \param x Distance (or cdf) input.
     * \return  The kernel value (linearly interpolated).
     */
    double evalBirthKernel(int s, double x) const;

    /**
     * \brief Evaluate the death (competition) kernel for species pair (s1, s2).
     * \param s1 The focal species (the one whose rate is affected).
     * \param s2 The "other" species (exerting the competition).
     * \param dist The distance between two particles.
     * \return The kernel value (linearly interpolated).
     */
    double evalDeathKernel(int s1, int s2, double dist) const;

    /**
     * \brief Place initial populations into the grid. Only the baseline death
     *        rates are set here; pairwise competition is added later via
     *        \c computeInitialDeathRates().
     *
     * \param initialCoords initialCoords[s] is a vector of positions for species s.
     */
    void placeInitialPopulations(
        const std::vector< std::vector< std::array<double,DIM> > > &initialCoords
    );

    /**
     * \brief Add pairwise interactions for every existing individual in each cell.
     *        This should be called after all initial particles are placed.
     */
    void computeInitialDeathRates();

    /**
     * \brief Creates a random unit vector in DIM dimensions (for birth/dispersal).
     *
     * Special case: in 1D, returns either +1 or -1 with 50% probability.
     *
     * \return A random direction of length ~1.0.
     */
    std::array<double, DIM> randomUnitVector(std::mt19937 & rng);

    /**
     * \brief Perform a single \b birth event in the simulation.
     *
     * Steps:
     *  1. Select a cell using discrete distribution over `cellBirthRate`.
     *  2. Select a species in that cell using `cellBirthRateBySpecies`.
     *  3. Select a random parent among `population[s]`.
     *  4. Sample a birth distance from `birth_x[s]/birth_y[s]` (ICDF).
     *  5. Pick a random direction, place child (respect boundaries).
     *  6. Update local cell caches (birth/death rates).
     *  7. Update pairwise competition for the new individual.
     */
    void spawn_random();

    /**
     * \brief Update pairwise interactions for a newly created particle.
     *
     * For the \p newIdx th particle of species \p sNew in cell \p cIdx:
     *   - Add dd[sNew][s2] * kernel contributions to the new particle.
     *   - Symmetrically, add dd[s2][sNew] contributions to neighbors.
     *
     * \param cIdx   The cell coordinate array where the new particle is placed.
     * \param sNew   Species index of the new particle.
     * \param newIdx Index of the new particle within cell.coords[sNew].
     */
    void updateInteractionsForNewParticle(
        const std::array<int,DIM> & cIdx,
        int sNew,
        int newIdx
    );

    /**
     * \brief Perform a single \b death event in the simulation.
     *
     * Steps:
     *  1. Select a cell using discrete distribution over `cellDeathRate`.
     *  2. Select a species using `cellDeathRateBySpecies`.
     *  3. Select a victim by discrete distribution over that species' `deathRates`.
     *  4. Remove the victim from the simulation, update local caches.
     *  5. Remove pairwise interaction the victim contributed to neighbors.
     */
    void kill_random();

    /**
     * \brief Remove all pairwise interactions contributed by a victim about
     *        to be removed.
     *
     * \param cellIndex  The flattened cell index of the victim.
     * \param sVictim    The species index of the victim.
     * \param victimIdx  The index of the victim in cell.coords[sVictim].
     */
    void removeInteractionsOfParticle(
        int cellIndex,
        int sVictim,
        int victimIdx
    );

    /**
     * \brief Execute one birth/death event.
     *
     * 1) Pick event type (birth vs. death) with probabilities
     *    proportional to `total_birth_rate` / `total_death_rate`.
     * 2) Advance simulation time by an exponential waiting time
     *    with parameter = (total_birth_rate + total_death_rate).
     * 3) Dispatch to \c spawn_random() or \c kill_random() accordingly.
     *
     * If `total_birth_rate + total_death_rate` is effectively 0, does nothing.
     */
    void make_event();
    void run_events(int events);
    void run_for(double time);
};

/// \brief Explicit instantiations (see SpatialBirthDeath.cpp)
extern template class Grid<1>;
extern template class Grid<2>;
extern template class Grid<3>;

extern template struct Cell<1>;
extern template struct Cell<2>;
extern template struct Cell<3>;

#endif // SPATIAL_BIRTH_DEATH_H
