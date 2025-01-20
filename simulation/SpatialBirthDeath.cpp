/**
 * \file SpatialBirthDeath.cpp
 * \brief Source file for the spatial birth-death point process simulator.
 *
 *
 * \date 2025-01-20
 */

#include "SpatialBirthDeath.h"

//============================================================
//  Implementation of Grid<DIM> members
//============================================================

/**
 * \brief Constructor implementation for the simulation grid.
 */
template<int DIM>
Grid<DIM>::Grid(int M_,
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
                // To ensure we visit at least the cell itself,
                // we can set a floor of 1:
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

template<int DIM>
int Grid<DIM>::flattenIdx(const std::array<int,DIM> &idx) const {
    int f = 0;
    int mul = 1;
    for (int dim=0; dim<DIM; dim++) {
        f += idx[dim]*mul;
        mul *= cell_count[dim];
    }
    return f;
}

template<int DIM>
int Grid<DIM>::wrapIndex(int i, int dim) const {
    if (!periodic) {
        // If not periodic, we simply do not wrap;
        // the caller is presumably controlling the range
        return i;
    }
    int n = cell_count[dim];
    if (i < 0)  i += n;
    if (i >= n) i -= n;
    return i;
}

template<int DIM>
Cell<DIM>& Grid<DIM>::cellAt(const std::array<int,DIM> &raw)
{
    std::array<int,DIM> w;
    for (int dim=0; dim<DIM; dim++) {
        w[dim] = wrapIndex(raw[dim], dim);
    }
    return cells[ flattenIdx(w) ];
}

template<int DIM>
double Grid<DIM>::evalBirthKernel(int s, double x) const {
    return linearInterpolate(birth_x[s], birth_y[s], x);
}

template<int DIM>
double Grid<DIM>::evalDeathKernel(int s1, int s2, double dist) const {
    return linearInterpolate(death_x[s1][s2], death_y[s1][s2], dist);
}

template<int DIM>
void Grid<DIM>::placeInitialPopulations(
    const std::vector< std::vector< std::array<double,DIM> > > &initialCoords
)
{
    for (int s = 0; s < M; s++) {
        for (auto &pos : initialCoords[s]) {
            // compute cell index
            std::array<int,DIM> cIdx;
            for (int dim=0; dim<DIM; dim++) {
                double coord = pos[dim];
                if (coord < 0) coord = 0;
                if (coord > area_length[dim]) coord = area_length[dim];
                int ic = (int)std::floor(coord * cell_count[dim] / area_length[dim]);
                if (ic == cell_count[dim]) ic--;
                cIdx[dim] = ic;
            }
            auto &cell = cellAt(cIdx);
            cell.coords[s].push_back(pos);

            // baseline death rate
            cell.deathRates[s].push_back(d[s]);
            cell.population[s] += 1;

            total_population++;

            // Update the cell's cached rates
            cell.cellDeathRateBySpecies[s] += d[s];
            cell.cellDeathRate += d[s];
            total_death_rate += d[s];

            cell.cellBirthRateBySpecies[s] += b[s];
            cell.cellBirthRate += b[s];
            total_birth_rate += b[s];
        }
    }
}

template<int DIM>
void Grid<DIM>::computeInitialDeathRates()
{
    // We already have baseline in each cell's deathRates[s][i],
    // and partial sums in cellDeathRateBySpecies[s].
    // Now we add pairwise interactions.

    for (int cellIndex = 0; cellIndex < total_num_cells; cellIndex++) {
        // unflatten
        std::array<int,DIM> cIdx;
        {
            int temp = cellIndex;
            for (int dim=0; dim<DIM; dim++) {
                cIdx[dim] = temp % cell_count[dim];
                temp /= cell_count[dim];
            }
        }
        Cell<DIM> &thisCell = cells[cellIndex];

        // For each species s1
        for (int s1 = 0; s1 < M; s1++) {
            // For each particle i of s1
            for (int i = 0; i < (int)thisCell.coords[s1].size(); i++) {
                auto &pos1 = thisCell.coords[s1][i];
                double &dr1 = thisCell.deathRates[s1][i];

                // for each s2
                for (int s2 = 0; s2 < M; s2++) {
                    auto cullRange = cull[s1][s2];
                    forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int,DIM> &nIdx)
                    {
                        Cell<DIM> & neighCell = cellAt(nIdx);
                        for (int j = 0; j < (int)neighCell.coords[s2].size(); j++) {
                            // skip self if same species & same cell & same index
                            if (&thisCell == &neighCell && s1 == s2 && i == j) {
                                return;
                            }
                            auto &pos2 = neighCell.coords[s2][j];
                            double dist = distancePeriodic<DIM>(pos1, pos2, area_length, periodic);
                            if (dist > cutoff[s1][s2]) {
                                return; // no effect beyond cutoff
                            }
                            double interaction = dd[s1][s2] * evalDeathKernel(s1, s2, dist);

                            // Add to i-th particle's death rate
                            dr1 += interaction;
                            thisCell.cellDeathRateBySpecies[s1] += interaction;
                            thisCell.cellDeathRate += interaction;
                            total_death_rate += interaction;
                        }
                    });
                }
            }
        }
    }
}

template<int DIM>
std::array<double, DIM> Grid<DIM>::randomUnitVector(std::mt19937 & rng)
{
    std::array<double, DIM> dir;
    if constexpr (DIM == 1) {
        // In 1D, pick +1 or -1
        std::uniform_real_distribution<double> u(0.0, 1.0);
        dir[0] = (u(rng) < 0.5) ? -1.0 : 1.0;
    }
    else {
        // For DIM >= 2, pick from normal distribution, then normalize
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
void Grid<DIM>::spawn_random()
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

    // 3) pick a random parent index
    int popS = parentCell.population[s];
    if (popS == 0) {
        // no parent -> cannot spawn
        return;
    }
    int parentIdx = std::uniform_int_distribution<int>(0, popS-1)(rng);
    auto & parentPos = parentCell.coords[s][parentIdx];

    // 4) sample birth radius (ICDF from birth kernel)
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double radius = linearInterpolate(birth_x[s], birth_y[s], u);

    // 5) pick random direction, multiply by radius
    auto dir = randomUnitVector(rng);
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
                // discard
                return;
            } else {
                double L = area_length[d];
                while (childPos[d] < 0.0)  childPos[d] += L;
                while (childPos[d] >= L)   childPos[d] -= L;
            }
        }
    }

    // 6) figure out child's cell
    std::array<int, DIM> childCellIdx;
    for (int d=0; d<DIM; d++) {
        int c = (int)std::floor(childPos[d] * cell_count[d] / area_length[d]);
        if (c == cell_count[d]) c--;
        childCellIdx[d] = c;
    }
    Cell<DIM> & childCell = cellAt(childCellIdx);

    // 7) Insert new individual
    childCell.coords[s].push_back(childPos);
    childCell.deathRates[s].push_back(d[s]);
    childCell.population[s]++;
    total_population++;

    // update local caches
    childCell.cellBirthRateBySpecies[s] += b[s];
    childCell.cellBirthRate += b[s];
    total_birth_rate += b[s];

    childCell.cellDeathRateBySpecies[s] += d[s];
    childCell.cellDeathRate += d[s];
    total_death_rate += d[s];

    // 8) Add pairwise interactions
    updateInteractionsForNewParticle(childCellIdx, s,
                                     (int)childCell.coords[s].size() - 1);
}

template<int DIM>
void Grid<DIM>::updateInteractionsForNewParticle(
    const std::array<int,DIM> & cIdx,
    int sNew,
    int newIdx
)
{
    Cell<DIM> & cell = cellAt(cIdx);
    auto & posNew = cell.coords[sNew][newIdx];

    // For each other species s2
    for (int s2 = 0; s2 < M; s2++) {
        auto cullRange = cull[sNew][s2];
        forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int,DIM> &nIdx)
        {
            Cell<DIM> & neighCell = cellAt(nIdx);
            for (int j=0; j<(int)neighCell.coords[s2].size(); j++) {
                // skip self
                if (&neighCell == &cell && s2 == sNew && j == newIdx) {
                    continue;
                }
                auto & pos2 = neighCell.coords[s2][j];
                double dist = distancePeriodic<DIM>(posNew, pos2, area_length, periodic);
                if (dist <= cutoff[sNew][s2]) {
                    double interaction = dd[sNew][s2] * evalDeathKernel(sNew, s2, dist);
                    cell.deathRates[sNew][newIdx] += interaction;
                    cell.cellDeathRateBySpecies[sNew] += interaction;
                    cell.cellDeathRate += interaction;
                    total_death_rate += interaction;

                    // symmetrical effect
                    double interaction2 = dd[s2][sNew] * evalDeathKernel(s2, sNew, dist);
                    neighCell.deathRates[s2][j] += interaction2;
                    neighCell.cellDeathRateBySpecies[s2] += interaction2;
                    neighCell.cellDeathRate += interaction2;
                    total_death_rate += interaction2;
                }
            }
        });
    }
}

template<int DIM>
void Grid<DIM>::kill_random()
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

    // 3) pick victim
    std::discrete_distribution<int> victimDist(
        cell.deathRates[s].begin(),
        cell.deathRates[s].end()
    );
    int victimIdx = victimDist(rng);
    double victimRate = cell.deathRates[s][victimIdx];

    // 4) remove from cell
    cell.population[s]--;
    total_population--;

    cell.cellDeathRateBySpecies[s] -= victimRate;
    cell.cellDeathRate -= victimRate;
    total_death_rate -= victimRate;

    cell.cellBirthRateBySpecies[s] -= b[s];
    cell.cellBirthRate -= b[s];
    total_birth_rate -= b[s];

    // remove pairwise interactions
    removeInteractionsOfParticle(cellIndex, s, victimIdx);

    // swap-and-pop
    int lastIdx = (int)cell.coords[s].size() - 1;
    cell.coords[s][victimIdx] = cell.coords[s][lastIdx];
    cell.deathRates[s][victimIdx] = cell.deathRates[s][lastIdx];
    cell.coords[s].pop_back();
    cell.deathRates[s].pop_back();
}

template<int DIM>
void Grid<DIM>::removeInteractionsOfParticle(
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

    // We'll remove the interactions contributed by victim
    for (int s2=0; s2<M; s2++) {
        auto cullRange = cull[sVictim][s2];
        forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int,DIM> &nIdx)
        {
            Cell<DIM> & neighCell = cellAt(nIdx);

            for (int j=0; j<(int)neighCell.coords[s2].size(); j++) {
                // skip the victim itself
                if (&neighCell == &cell && s2 == sVictim && j == victimIdx) {
                    continue;
                }
                auto & pos2 = neighCell.coords[s2][j];
                double dist = distancePeriodic<DIM>(posVictim, pos2, area_length, periodic);
                if (dist <= cutoff[sVictim][s2]) {
                    double interaction = dd[sVictim][s2] * evalDeathKernel(sVictim, s2, dist);
                    neighCell.deathRates[s2][j] -= interaction;
                    neighCell.cellDeathRateBySpecies[s2] -= interaction;
                    neighCell.cellDeathRate -= interaction;
                    total_death_rate -= interaction;

                    double interaction2 = dd[s2][sVictim] * evalDeathKernel(s2, sVictim, dist);
                    cell.cellDeathRateBySpecies[sVictim] -= interaction2;
                    cell.cellDeathRate -= interaction2;
                    total_death_rate -= interaction2;
                }
            }
        });
    }
}

template<int DIM>
void Grid<DIM>::make_event()
{
    double sumRate = total_birth_rate + total_death_rate;
    if (sumRate < 1e-12) {
        // no event possible
        return;
    }
    event_count++;

    // 1) Advance time
    std::exponential_distribution<double> expDist(sumRate);
    time += expDist(rng);

    // 2) Decide birth or death
    double r = std::uniform_real_distribution<double>(0.0, sumRate)(rng);
    bool isBirth = (r < total_birth_rate);

    if (isBirth) {
        spawn_random();
    } else {
        kill_random();
    }
}

//============================================================
//  Explicit template instantiations
//============================================================
template class Grid<1>;
template class Grid<2>;
template class Grid<3>;
