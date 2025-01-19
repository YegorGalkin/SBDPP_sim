#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <stdexcept>

//============================================================
// 1) Utility: linear interpolation (spline placeholders, etc.)
//============================================================
inline double linearInterpolate(const std::vector<double> &xVals,
                                const std::vector<double> &yVals,
                                double x)
{
    if (x <= xVals.front()) return yVals.front();
    if (x >= xVals.back())  return yVals.back();
    auto it = std::lower_bound(xVals.begin(), xVals.end(), x);
    size_t idx = (it - xVals.begin());
    if (idx == 0) return yVals[0];
    double x1 = xVals[idx-1];
    double x2 = xVals[idx];
    double y1 = yVals[idx-1];
    double y2 = yVals[idx];
    double t = (x - x1)/(x2 - x1);
    return y1 + t*(y2 - y1);
}

//============================================================
// 2) Recursive neighbor iteration
//============================================================
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

template<int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &center,
                  const std::array<int, DIM> &cull,
                  const FUNC &func)
{
    std::array<int, DIM> temp;
    forNeighborsRecur<DIM>(center, cull, temp, 0, func);
}

//============================================================
// 3) Periodic distance
//============================================================
template<int DIM>
double distancePeriodic(const std::array<double,DIM> &a,
                        const std::array<double,DIM> &b,
                        const std::array<double,DIM> &length,
                        bool periodic)
{
    double dist2 = 0.0;
    for (int d=0; d<DIM; d++) {
        double diff = a[d] - b[d];
        if (periodic) {
            double L = length[d];
            if      (diff >  0.5*L) diff -= L;
            else if (diff < -0.5*L) diff += L;
        }
        dist2 += diff*diff;
    }
    return std::sqrt(dist2);
}

//============================================================
// 4) Cell data structure
//
//    - For each species s:
//       coords[s][i] = coordinate of i-th particle
//       deathRates[s][i] = individual's death rate
//       population[s] = # of individuals
//
//    - Summations (for fast discrete sampling):
//       cellBirthRateBySpecies[s]
//       cellDeathRateBySpecies[s]
//       cellBirthRate  = sum_s cellBirthRateBySpecies[s]
//       cellDeathRate  = sum_s cellDeathRateBySpecies[s]
//============================================================
template<int DIM>
struct Cell
{
    // Per‐species arrays
    std::vector< std::vector< std::array<double, DIM> > > coords;      // coords[s]
    std::vector< std::vector<double> >                    deathRates;  // deathRates[s]
    std::vector<int>                                      population;  // population[s]

    // Cached sums for discrete sampling
    std::vector<double> cellBirthRateBySpecies;  // b[s]*population[s], etc.
    std::vector<double> cellDeathRateBySpecies;  // sum of deathRates[s]

    double cellBirthRate;  // sum(cellBirthRateBySpecies[s])
    double cellDeathRate;  // sum(cellDeathRateBySpecies[s])

    Cell() : cellBirthRate(0.0), cellDeathRate(0.0)
    {}

    void initSpecies(int M) {
        coords.resize(M);
        deathRates.resize(M);
        population.resize(M, 0);
        cellBirthRateBySpecies.resize(M, 0.0);
        cellDeathRateBySpecies.resize(M, 0.0);
    }
};

//============================================================
// 5) Grid class
//
//    - M species
//    - b[s], d[s], dd[s1][s2]
//    - birth kernel arrays birth_x[s], birth_y[s]
//    - death kernel arrays death_x[s1][s2], death_y[s1][s2]
//    - cutoff[s1][s2], cull[s1][s2][DIM]
//    - cells[]: each cell is a Cell<DIM>
//    - total_birth_rate, total_death_rate for discrete event selection
//============================================================
template<int DIM>
class Grid
{
public:
    // Simulation parameters
    int M; // number of species
    std::array<double, DIM> area_length;
    std::array<int,    DIM> cell_count;
    bool periodic;

    // Per-species baseline rates
    std::vector<double> b;     // birth rate for each species
    std::vector<double> d;     // baseline death rate for each species

    // Pairwise interaction
    std::vector< std::vector<double> > dd;

    // Kernels
    std::vector< std::vector<double> > birth_x; // birth_x[s]
    std::vector< std::vector<double> > birth_y; // birth_y[s]
    std::vector< std::vector< std::vector<double> > > death_x; // death_x[s1][s2]
    std::vector< std::vector< std::vector<double> > > death_y; // death_y[s1][s2]

    // Cutoff, cull arrays
    std::vector< std::vector<double> > cutoff;                // cutoff[s1][s2]
    std::vector< std::vector< std::array<int,DIM> > > cull;   // cull[s1][s2][dim]

    // Grid cells
    std::vector< Cell<DIM> > cells;
    int total_num_cells;

    // Summaries
    double total_birth_rate;
    double total_death_rate;
    int    total_population;

    // RNG, time
    std::mt19937 rng;
    double time;
    int    event_count;
    std::chrono::system_clock::time_point init_time;
    double realtime_limit;
    bool   realtime_limit_reached;

public:
    //====================================================
    // Constructor
    //====================================================
    Grid(int M_,
         const std::array<double,DIM> &areaLen,
         const std::array<int,DIM> &cellCount_,
         bool isPeriodic,
         // b,d,dd
         const std::vector<double> &birthRates,    // size M
         const std::vector<double> &deathRates,    // size M
         const std::vector<double> &ddMatrix,      // size M*M
         // birth kernel
         const std::vector< std::vector<double> > &birthX,
         const std::vector< std::vector<double> > &birthY,
         // death kernel
         const std::vector< std::vector< std::vector<double> > > &deathX_,
         const std::vector< std::vector< std::vector<double> > > &deathY_,
         // cutoff
         const std::vector<double> &cutoffs, // size M*M
         int seed,
         double rtimeLimit
    )
    : M(M_), area_length(areaLen), cell_count(cellCount_),
      periodic(isPeriodic), total_birth_rate(0.0), total_death_rate(0.0),
      total_population(0), rng(seed), time(0.0), event_count(0),
      realtime_limit(rtimeLimit), realtime_limit_reached(false)
    {
        init_time = std::chrono::system_clock::now();

        // 1) Copy rates
        b = birthRates;
        d = deathRates;

        dd.resize(M);
        for (int s1=0; s1<M; s1++) {
            dd[s1].resize(M);
            for (int s2=0; s2<M; s2++) {
                dd[s1][s2] = ddMatrix[s1*M + s2];
            }
        }

        // 2) Copy birth kernel
        birth_x = birthX;
        birth_y = birthY;

        // 3) Copy death kernel
        death_x.resize(M);
        death_y.resize(M);
        for (int s1=0; s1<M; s1++) {
            death_x[s1].resize(M);
            death_y[s1].resize(M);
            for (int s2=0; s2<M; s2++) {
                death_x[s1][s2] = deathX_[s1][s2];
                death_y[s1][s2] = deathY_[s1][s2];
            }
        }

        // 4) cutoff & cull
        cutoff.resize(M);
        for (int s1=0; s1<M; s1++) {
            cutoff[s1].resize(M);
            for (int s2=0; s2<M; s2++) {
                cutoff[s1][s2] = cutoffs[s1*M + s2];
            }
        }
        // build cull
        cull.resize(M);
        for (int s1=0; s1<M; s1++) {
            cull[s1].resize(M);
            for (int s2=0; s2<M; s2++) {
                for (int dim=0; dim<DIM; dim++) {
                    double cellSize = area_length[dim]/cell_count[dim];
                    int needed = (int)std::ceil(cutoff[s1][s2]/cellSize);
                    cull[s1][s2][dim] = std::max(needed, 1);
                }
            }
        }

        // 5) total_num_cells
        {
            int prod = 1;
            for (int dim=0; dim<DIM; dim++) {
                prod *= cell_count[dim];
            }
            total_num_cells = prod;
        }

        // 6) Allocate cells
        cells.resize(total_num_cells);
        for (auto &c : cells) {
            c.initSpecies(M);
        }
    }

    //====================================================
    // Flatten/unflatten
    //====================================================
    int flattenIdx(const std::array<int,DIM> &idx) const {
        int f = 0;
        int mul = 1;
        for (int dim=0; dim<DIM; dim++) {
            f += idx[dim]*mul;
            mul *= cell_count[dim];
        }
        return f;
    }

    int wrapIndex(int i, int dim) const {
        if (!periodic) return i;
        int n = cell_count[dim];
        if (i<0)    i += n;
        if (i>=n)   i -= n;
        return i;
    }

    Cell<DIM>& cellAt(const std::array<int,DIM> &raw) {
        std::array<int,DIM> w;
        for (int dim=0; dim<DIM; dim++) {
            w[dim] = wrapIndex(raw[dim], dim);
        }
        return cells[ flattenIdx(w) ];
    }

    //====================================================
    // Evaluate kernels
    //====================================================
    double evalBirthKernel(int s, double x) const {
        return linearInterpolate(birth_x[s], birth_y[s], x);
    }
    double evalDeathKernel(int s1, int s2, double dist) const {
        return linearInterpolate(death_x[s1][s2], death_y[s1][s2], dist);
    }

    //====================================================
    // Place initial populations
    //   initialCoords[s] => vector of positions for species s
    //
    // We only add baseline death, then we do neighbor
    // interaction in computeInitialDeathRates().
    //====================================================
    void placeInitialPopulations(
        const std::vector< std::vector< std::array<double,DIM> > > &initialCoords
    )
    {
        for (int s=0; s<M; s++) {
            for (auto &pos : initialCoords[s]) {
                // compute cell index
                std::array<int,DIM> cIdx;
                for (int dim=0; dim<DIM; dim++) {
                    double coord = pos[dim];
                    if (coord<0) coord=0;
                    if (coord>area_length[dim]) coord=area_length[dim];
                    int ic = (int)std::floor(coord*cell_count[dim]/area_length[dim]);
                    if (ic==cell_count[dim]) ic--;
                    cIdx[dim] = ic;
                }
                auto &cell = cellAt(cIdx);
                cell.coords[s].push_back(pos);
                cell.deathRates[s].push_back(d[s]); // baseline
                cell.population[s]+=1;

                total_population++;
                // We'll add the baseline to cellDeathRateBySpecies, but not interaction yet
                cell.cellDeathRateBySpecies[s] += d[s];
                cell.cellDeathRate += d[s];
                total_death_rate += d[s];

                // For births, a simple model: cellBirthRateBySpecies[s] = b[s]*pop[s]
                // We'll just add b[s] now that we have 1 new individual
                cell.cellBirthRateBySpecies[s] += b[s];
                cell.cellBirthRate += b[s];
                total_birth_rate += b[s];
            }
        }
    }

    //====================================================
    // Build pairwise interactions for each cell
    //  (adds dd[s1][s2]*f(dist) to death rates)
    //  Then update cellDeathRateBySpecies[s] accordingly
    //  => caching for discrete sampling
    //====================================================
    void computeInitialDeathRates()
    {
        // We already have baseline in each cell's `deathRates[s]`,
        // and partial sums in `cellDeathRateBySpecies[s]`.
        // Now we add pairwise interactions.

        for (int cellIndex=0; cellIndex<total_num_cells; cellIndex++) {
            // unflatten
            std::array<int,DIM> cIdx;
            {
                int temp = cellIndex;
                for (int dim=0; dim<DIM; dim++) {
                    cIdx[dim] = temp % cell_count[dim];
                    temp      = temp / cell_count[dim];
                }
            }
            Cell<DIM> &thisCell = cells[cellIndex];

            // For each species s1
            for (int s1=0; s1<M; s1++) {
                // For each particle i of s1
                for (int i=0; i<(int)thisCell.coords[s1].size(); i++) {
                    auto &pos1 = thisCell.coords[s1][i];
                    double &dr1 = thisCell.deathRates[s1][i];

                    // for each s2
                    for (int s2=0; s2<M; s2++) {
                        auto cullRange = cull[s1][s2];

                        forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int,DIM> &nIdx){
                            Cell<DIM> & neighCell = cellAt(nIdx);
                            for (int j=0; j<(int)neighCell.coords[s2].size(); j++) {
                                // skip self
                                if (&thisCell == &neighCell && s1==s2 && i==j) {
                                    return; // continue
                                }
                                auto &pos2 = neighCell.coords[s2][j];
                                double dist = distancePeriodic<DIM>(pos1, pos2, area_length, periodic);
                                if (dist>cutoff[s1][s2]) return;
                                double interaction = dd[s1][s2]* evalDeathKernel(s1, s2, dist);
                                dr1 += interaction;
                                thisCell.cellDeathRateBySpecies[s1]+= interaction;
                                thisCell.cellDeathRate += interaction;
                                total_death_rate += interaction;
                            }
                        }); // end forNeighbors
                    } // end s2
                } // end i
            } // end s1
        } // end cellIndex
    }


    // Return a random unit vector in DIM dimensions
    // For DIM=1, "unit vector" is either +1 or -1 with 50% chance.
    template<int DIM>
    std::array<double, DIM> randomUnitVector(std::mt19937 & rng)
    {
        std::array<double, DIM> dir;
        if constexpr (DIM == 1) {
            // random sign
            std::uniform_real_distribution<double> u(0.0, 1.0);
            dir[0] = (u(rng) < 0.5) ? -1.0 : 1.0;
        }
        else {
            // pick each component from normal distribution, then normalize
            std::normal_distribution<double> gauss(0.0, 1.0);
            double sumSq = 0.0;
            for (int d=0; d<DIM; d++) {
                double val = gauss(rng);
                dir[d] = val;
                sumSq += val*val;
            }
            double inv = 1.0 / std::sqrt(sumSq + 1e-14);
            for (int d=0; d<DIM; d++) {
                dir[d] *= inv;
            }
        }
        return dir;
    }

    template<int DIM>
    void spawn_random()
    {
        // 1) pick cell by discrete distribution of cellBirthRate
        std::vector<double> cellRateVec(total_num_cells);
        for (int i=0; i<total_num_cells; i++) {
            cellRateVec[i] = cells[i].cellBirthRate;
        }
        std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
        int cellIndex = cellDist(rng);
        Cell<DIM> & parentCell = cells[cellIndex];

        // 2) pick species by cellBirthRateBySpecies
        std::discrete_distribution<int> spDist(
            parentCell.cellBirthRateBySpecies.begin(),
            parentCell.cellBirthRateBySpecies.end()
        );
        int s = spDist(rng);

        // 3) pick parent index (uniform among population[s])
        int popS = parentCell.population[s];
        if (popS == 0) {
            // no parent -> cannot spawn
            return;
        }
        int parentIdx = std::uniform_int_distribution<int>(0, popS-1)(rng);
        auto & parentPos = parentCell.coords[s][parentIdx];

        // 4) sample birth radius from birth kernel
        //    For example, we assume birth_x[s] & birth_y[s] define an
        //    "inverse radial CDF" from 0..1 => 0..maxRadius
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        double radius = linearInterpolate(birth_x[s], birth_y[s], u);

        // 5) pick random direction
        auto dir = randomUnitVector<DIM>(rng);

        // multiply by radius
        for (int d=0; d<DIM; d++) {
            dir[d] *= radius;
        }

        // new position = parent position + dir
        std::array<double, DIM> childPos;
        for (int d=0; d<DIM; d++) {
            childPos[d] = parentPos[d] + dir[d];
            // boundary check
            if (childPos[d] < 0.0 || childPos[d] > area_length[d]) {
                if (!periodic) {
                    // If non‐periodic, we can either discard or clamp
                    return; // discard spawn
                } else {
                    // wrap
                    double L = area_length[d];
                    // e.g. childPos[d] = fmod(childPos[d], L);
                    // but watch negative remainders:
                    while (childPos[d] < 0.0)  childPos[d] += L;
                    while (childPos[d] >= L)   childPos[d] -= L;
                }
            }
        }

        // 6) figure out which cell the child belongs to
        std::array<int, DIM> childCellIdx;
        for (int d=0; d<DIM; d++) {
            int c = (int)std::floor(childPos[d] * cell_count[d] / area_length[d]);
            if (c == cell_count[d]) c--;
            childCellIdx[d] = c;
        }
        Cell<DIM> & childCell = cellAt(childCellIdx);

        // 7) Insert new individual into childCell
        childCell.coords[s].push_back(childPos);
        // baseline death rate = d[s]
        childCell.deathRates[s].push_back(d[s]);
        childCell.population[s]++;

        total_population++;

        // Update local cell's birth/death caches
        //  - for births, we add +b[s] to cellBirthRate
        //  - for death, we add +d[s]
        childCell.cellBirthRateBySpecies[s] += b[s];
        childCell.cellBirthRate += b[s];
        total_birth_rate += b[s];

        childCell.cellDeathRateBySpecies[s] += d[s];
        childCell.cellDeathRate += d[s];
        total_death_rate += d[s];

        // 8) Add pairwise interactions *only* for the newly created particle
        //    with neighbors (including the cell itself).
        //    We'll define a small helper:
        updateInteractionsForNewParticle(childCellIdx, s,
                                         (int)childCell.coords[s].size() - 1);
    }

    //------------------------------------------------------------
    // Add pairwise interactions for the newly created particle
    // (with index newIdx in species s), looking at neighbors
    //------------------------------------------------------------
    template<int DIM>
    void updateInteractionsForNewParticle(
        const std::array<int,DIM> & cIdx,
        int sNew,
        int newIdx
    )
    {
        // position of the new particle
        Cell<DIM> & cell = cellAt(cIdx);
        auto & posNew = cell.coords[sNew][newIdx];

        // For each s2, we only need cull[sNew][s2] range
        for (int s2=0; s2<M; s2++) {
            auto cullRange = cull[sNew][s2];

            // Iterate neighbors
            forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int,DIM> &nIdx){
                Cell<DIM> & neighCell = cellAt(nIdx);

                // For each individual of s2
                for (int j=0; j<(int)neighCell.coords[s2].size(); j++) {
                    // If sNew == s2 and &cell==&neighCell and j==newIdx => skip self
                    if (&neighCell == &cell && s2==sNew && j==newIdx) {
                        continue;
                    }

                    auto & pos2 = neighCell.coords[s2][j];
                    double dist = distancePeriodic<DIM>(posNew, pos2, area_length, periodic);
                    if (dist <= cutoff[sNew][s2]) {
                        double interaction = dd[sNew][s2] * evalDeathKernel(sNew, s2, dist);

                        // Add this to the new particle's deathRates
                        cell.deathRates[sNew][newIdx] += interaction;
                        cell.cellDeathRateBySpecies[sNew] += interaction;
                        cell.cellDeathRate += interaction;
                        total_death_rate += interaction;

                        // Also symmetrical effect: the new particle influences that existing neighbor
                        // => we add dd[s2][sNew]* kernel to the neighbor's death rate
                        if (!(sNew==s2 && &neighCell==&cell && j==newIdx)) {
                            double interaction2 = dd[s2][sNew]* evalDeathKernel(s2, sNew, dist);
                            neighCell.deathRates[s2][j] += interaction2;
                            neighCell.cellDeathRateBySpecies[s2] += interaction2;
                            neighCell.cellDeathRate += interaction2;
                            total_death_rate += interaction2;
                        }
                    }
                }
            });
        }
    }

    template<int DIM>
    void kill_random()
    {
        // 1) pick cell
        std::vector<double> cellRateVec(total_num_cells);
        for (int i=0; i<total_num_cells; i++) {
            cellRateVec[i] = cells[i].cellDeathRate;
        }
        std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
        int cellIndex = cellDist(rng);
        Cell<DIM> & cell = cells[cellIndex];

        // 2) pick species
        std::discrete_distribution<int> spDist(
            cell.cellDeathRateBySpecies.begin(),
            cell.cellDeathRateBySpecies.end()
        );
        int s = spDist(rng);

        if (cell.population[s] == 0) {
            // no one to kill
            return;
        }

        // 3) pick victim by discrete distribution on that species's deathRates
        std::discrete_distribution<int> victimDist(
            cell.deathRates[s].begin(),
            cell.deathRates[s].end()
        );
        int victimIdx = victimDist(rng);
        double victimRate = cell.deathRates[s][victimIdx];

        // 4) remove from the cell
        //    first subtract baseline from caches
        cell.population[s]--;
        total_population--;

        cell.cellDeathRateBySpecies[s] -= victimRate;
        cell.cellDeathRate -= victimRate;
        total_death_rate -= victimRate;

        // each individual contributed +b[s] to birth rate
        cell.cellBirthRateBySpecies[s] -= b[s];
        cell.cellBirthRate -= b[s];
        total_birth_rate -= b[s];

        // We also need to remove pairwise interaction from neighbors
        // The victim’s own deathRates[s][victimIdx] includes baseline + interaction.
        // We'll do an incremental neighbor update:

        removeInteractionsOfParticle(cellIndex, s, victimIdx);

        // Now do the "swap-and-pop" to remove from coords[s], deathRates[s]
        int lastIdx = (int)cell.coords[s].size() - 1;
        cell.coords[s][victimIdx] = cell.coords[s][lastIdx];
        cell.deathRates[s][victimIdx] = cell.deathRates[s][lastIdx];
        cell.coords[s].pop_back();
        cell.deathRates[s].pop_back();
    }

    //------------------------------------------------------------
    // removeInteractionsOfParticle:
    //   subtract the pairwise interactions contributed by
    //   the victim from the neighbors
    //------------------------------------------------------------
    template<int DIM>
    void removeInteractionsOfParticle(
        int cellIndex,
        int sVictim,
        int victimIdx
    )
    {
        // unflatten
        std::array<int,DIM> cIdx;
        {
            int tmp = cellIndex;
            for (int dim=0; dim<DIM; dim++) {
                cIdx[dim] = tmp % cell_count[dim];
                tmp /= cell_count[dim];
            }
        }

        Cell<DIM> & cell = cells[cellIndex];
        auto & posVictim = cell.coords[sVictim][victimIdx];
        double victimTotalRate = cell.deathRates[sVictim][victimIdx];
        // This includes baseline + sum of interactions from neighbors.

        // We'll do partial neighbor iteration. For each s2, we look in cull[sVictim][s2].
        for (int s2=0; s2<M; s2++) {
            auto cullRange = cull[sVictim][s2];

            forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int,DIM> &nIdx){
                Cell<DIM> & neighCell = cellAt(nIdx);

                // We'll walk through each particle in s2, check if distance < cutoff => if so,
                // subtract the interaction from that neighbor’s death rate and from total_death_rate, etc.
                for (int j=0; j<(int)neighCell.coords[s2].size(); j++) {
                    // skip self
                    if (&neighCell == &cell && s2==sVictim && j==victimIdx) {
                        continue;
                    }
                    auto & pos2 = neighCell.coords[s2][j];
                    double dist = distancePeriodic<DIM>(posVictim, pos2, area_length, periodic);
                    if (dist <= cutoff[sVictim][s2]) {
                        // the victim contributed dd[sVictim][s2]* kernel to pos2
                        double interaction = dd[sVictim][s2] * evalDeathKernel(sVictim, s2, dist);

                        neighCell.deathRates[s2][j] -= interaction;
                        neighCell.cellDeathRateBySpecies[s2] -= interaction;
                        neighCell.cellDeathRate -= interaction;
                        total_death_rate -= interaction;

                        // symmetrical direction: the neighbor contributed dd[s2][sVictim]* kernel to victim
                        // we need to remove that portion from the victim’s cell / total as well:
                        double interaction2 = dd[s2][sVictim]* evalDeathKernel(s2, sVictim, dist);
                        // The victim’s own rate is about to be removed entirely anyway, but for correctness:
                        cell.cellDeathRateBySpecies[sVictim] -= interaction2;
                        cell.cellDeathRate -= interaction2;
                        total_death_rate -= interaction2;
                    }
                }
            });
        }
    }

    //====================================================
    // Example of make_event:
    //   1) Pick event type by total_birth_rate / (total_birth_rate + total_death_rate)
    //   2a) If birth:
    //       - pick a cell by discrete distribution on cellBirthRate
    //       - pick species by cellBirthRateBySpecies
    //       - pick random parent among population[s]
    //       - spawn
    //   2b) If death:
    //       - pick a cell by discrete distribution on cellDeathRate
    //       - pick species by cellDeathRateBySpecies
    //       - pick random victim by discrete distribution on that species's deathRates
    //       - kill
    //   3) Update local cell + total caches
    //====================================================
    void make_event()
    {
        double sumRate = total_birth_rate + total_death_rate;
        if (sumRate < 1e-12) {
            // no event possible
            return;
        }
        event_count++;

        // 1) Advance time by exponential
        std::exponential_distribution<double> expDist(sumRate);
        time += expDist(rng);

        // 2) Decide if birth or death
        double r = std::uniform_real_distribution<double>(0.0, sumRate)(rng);
        bool isBirth = (r < total_birth_rate);

        if (isBirth) {
            spawn_random()
        } else {
            kill_random()
        }
    }
};
