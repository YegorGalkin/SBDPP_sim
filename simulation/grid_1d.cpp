#include "grid_1d.h"
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>

// Constructor implementation
Grid1D::Grid1D(double area_length_x_, int cell_count_x_,
               double b_, double d_, double dd_, int seed_,
               const std::vector<double>& initial_population,
               const std::vector<double>& death_values,
               double death_cutoff_r_,
               const std::vector<double>& birth_values,
               bool periodic_, double realtime_limit_)
    : area_length_x(area_length_x_),
      cell_count_x(cell_count_x_),
      b(b_), d(d_), dd(dd_),
      seed(seed_),
      initial_population_x(initial_population),
      death_y(death_values),
      birth_inverse_rcdf_y(birth_values),
      death_cutoff_r(death_cutoff_r_),
      periodic(periodic_),
      realtime_limit(realtime_limit_),
      realtime_limit_reached(false),
      total_population(0),
      total_death_rate(0),
      event_count(0),
      time(0.0) {

    rng = std::mt19937(seed);

    death_spline_nodes = static_cast<int>(death_y.size());
    death_step = death_cutoff_r / (death_spline_nodes - 1);

    birth_inverse_rcdf_nodes = static_cast<int>(birth_inverse_rcdf_y.size());
    birth_inverse_rcdf_step = 1.0 / (birth_inverse_rcdf_nodes - 1);

    // Precompute x values for death and birth kernels
    death_x = std::vector<double>(death_spline_nodes);
    for (int i = 0; i < death_spline_nodes; ++i) {
        death_x[i] = i * death_step;
    }

    birth_inverse_rcdf_x = std::vector<double>(birth_inverse_rcdf_nodes);
    for (int i = 0; i < birth_inverse_rcdf_nodes; ++i) {
        birth_inverse_rcdf_x[i] = i * birth_inverse_rcdf_step;
    }

    init_time = std::chrono::system_clock::now();

    // Calculate the number of cells to check around for death interaction
    cull_x = std::max(static_cast<int>(std::ceil(death_cutoff_r / (area_length_x / cell_count_x))), 3);

    // Initialize death rates
    Initialize_death_rates();

    // Compute the total death rate
    total_death_rate = std::accumulate(cell_death_rates.begin(), cell_death_rates.end(), 0.0);
}

// Generic linear interpolation function
double Grid1D::linear_interpolate(const std::vector<double>& xgdat, const std::vector<double>& gdat, double x) {
    auto i = std::lower_bound(xgdat.begin(), xgdat.end(), x); // Nearest-above index
    size_t k = i - xgdat.begin();

    size_t l = k ? k - 1 : 0; // Nearest-below index

    // Linear interpolation formula
    double x1 = xgdat[l], x2 = xgdat[k];
    double y1 = gdat[l], y2 = gdat[k];

    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

// Evaluate the death kernel using linear interpolation
double Grid1D::death_kernel(double at) {
    return linear_interpolate(death_x, death_y, at);
}

// Evaluate the birth inverse RCDF kernel using linear interpolation
double Grid1D::birth_inverse_rcdf_kernel(double at) {
    return linear_interpolate(birth_inverse_rcdf_x, birth_inverse_rcdf_y, at);
}

// Access specific cell
Cell_1d& Grid1D::cell_at(int i) {
    if (periodic) {
        if (i < 0)
            i += cell_count_x;
        if (i >= cell_count_x)
            i -= cell_count_x;
    }
    return cells[i];
}

// Access specific cell death rate
double& Grid1D::cell_death_rate_at(int i) {
    if (periodic) {
        if (i < 0)
            i += cell_count_x;
        if (i >= cell_count_x)
            i -= cell_count_x;
    }
    return cell_death_rates[i];
}

// Access specific cell population
int& Grid1D::cell_population_at(int i) {
    if (periodic) {
        if (i < 0)
            i += cell_count_x;
        if (i >= cell_count_x)
            i -= cell_count_x;
    }
    return cell_population[i];
}

// Get x coordinates at a specific cell
std::vector<double> Grid1D::get_x_coords_at_cell(int i) {
    return cells[i].coords_x;
}

// Get death rates at a specific cell
std::vector<double> Grid1D::get_death_rates_at_cell(int i) {
    return cells[i].death_rates;
}

// Get all x coordinates
std::vector<double> Grid1D::get_all_x_coords() {
    std::vector<double> result;
    for (const auto& cell : cells) {
        if (!cell.coords_x.empty())
            result.insert(result.end(), cell.coords_x.begin(), cell.coords_x.end());
    }
    return result;
}

// Get all death rates
std::vector<double> Grid1D::get_all_death_rates() {
    std::vector<double> result;
    for (const auto& cell : cells) {
        if (!cell.coords_x.empty())
            result.insert(result.end(), cell.death_rates.begin(), cell.death_rates.end());
    }
    return result;
}

// Initialize death rates
void Grid1D::Initialize_death_rates() {
    for (int i = 0; i < cell_count_x; i++) {
        cells.push_back(Cell_1d());
        cell_death_rates.push_back(0);
        cell_population.push_back(0);
    }

    for (double x_coord : initial_population_x) {
        if (x_coord < 0 || x_coord > area_length_x)
            continue;

        int i = static_cast<int>(floor(x_coord * cell_count_x / area_length_x));
        if (i == cell_count_x)
            i--;

        cell_at(i).coords_x.push_back(x_coord);
        cell_at(i).death_rates.push_back(d);
        cell_death_rate_at(i) += d;
        total_death_rate += d;

        cell_population_at(i)++;
        total_population++;
    }

    for (int i = 0; i < cell_count_x; i++) {
        for (int k = 0; k < cell_population_at(i); k++) {
            for (int n = i - cull_x; n < i + cull_x + 1; n++) {
                if (!periodic && (n < 0 || n >= cell_count_x))
                    continue;

                for (int p = 0; p < cell_population_at(n); p++) {
                    if (i == n && k == p)
                        continue;

                    double distance;

                    if (periodic) {
                        if (n < 0) {
                            distance = abs(cell_at(i).coords_x[k] - cell_at(n).coords_x[p] + area_length_x);
                        } else if (n >= cell_count_x) {
                            distance = abs(cell_at(i).coords_x[k] - cell_at(n).coords_x[p] - area_length_x);
                        } else {
                            distance = abs(cell_at(i).coords_x[k] - cell_at(n).coords_x[p]);
                        }
                    } else {
                        distance = abs(cell_at(i).coords_x[k] - cell_at(n).coords_x[p]);
                    }

                    if (distance > death_cutoff_r)
                        continue;

                    double interaction = dd * death_kernel(distance);

                    cell_at(i).death_rates[k] += interaction;
                    cell_death_rate_at(i) += interaction;
                    total_death_rate += interaction;
                }
            }
        }
    }
}

void Grid1D::kill_random() {
    if (total_population == 1) {
        total_population--;
        return;
    }

    std::discrete_distribution<> cell_dist(cell_death_rates.begin(), cell_death_rates.end());
    int cell_death_index = cell_dist(rng);

    std::discrete_distribution<> in_cell_dist(cells[cell_death_index].death_rates.begin(), cells[cell_death_index].death_rates.end());
    int in_cell_death_index = in_cell_dist(rng);

    Cell_1d& death_cell = cells[cell_death_index];
    int cell_death_x = cell_death_index;

    for (int i = cell_death_x - cull_x; i < cell_death_x + cull_x + 1; i++) {
        if (!periodic && (i < 0 || i >= cell_count_x))
            continue;
        for (int k = 0; k < cell_population_at(i); k++) {
            if (i == cell_death_x && k == in_cell_death_index)
                continue;

            double distance;
            if (periodic) {
                if (i < 0) {
                    distance = abs(cell_at(cell_death_x).coords_x[in_cell_death_index] -
                                   cell_at(i).coords_x[k] + area_length_x);
                } else if (i >= cell_count_x) {
                    distance = abs(cell_at(cell_death_x).coords_x[in_cell_death_index] -
                                   cell_at(i).coords_x[k] - area_length_x);
                } else {
                    distance = abs(cell_at(cell_death_x).coords_x[in_cell_death_index] - cell_at(i).coords_x[k]);
                }
            } else {
                distance = abs(cell_at(cell_death_x).coords_x[in_cell_death_index] - cell_at(i).coords_x[k]);
            }

            if (distance > death_cutoff_r)
                continue;

            double interaction = dd * death_kernel(distance);

            cell_at(i).death_rates[k] -= interaction;
            cell_death_rate_at(i) -= interaction;
            cell_death_rate_at(cell_death_x) -= interaction;
            total_death_rate -= 2 * interaction;
        }
    }

    cell_death_rates[cell_death_index] -= d;
    total_death_rate -= d;
    cell_population[cell_death_index]--;
    total_population--;

    death_cell.death_rates[in_cell_death_index] = death_cell.death_rates.back();
    death_cell.coords_x[in_cell_death_index] = death_cell.coords_x.back();
    death_cell.death_rates.pop_back();
    death_cell.coords_x.pop_back();
}

void Grid1D::spawn_random() {
    std::discrete_distribution<> cell_dist(cell_population.begin(), cell_population.end());
    int cell_index = cell_dist(rng);

    std::uniform_int_distribution<> event_dist(0, cell_population[cell_index] - 1);
    int event_index = event_dist(rng);

    Cell_1d& parent_cell = cells[cell_index];

    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    double x_coord_new = parent_cell.coords_x[event_index] +
                         birth_inverse_rcdf_kernel(uniform_dist(rng)) * (uniform_dist(rng) > 0.5 ? 1 : -1);

    if (x_coord_new < 0 || x_coord_new > area_length_x) {
        if (!periodic) {
            return;
        } else {
            if (x_coord_new < 0)
                x_coord_new += area_length_x;
            if (x_coord_new > area_length_x)
                x_coord_new -= area_length_x;
        }
    }

    int new_i = static_cast<int>(floor(x_coord_new * cell_count_x / area_length_x));
    if (new_i == cell_count_x)
        new_i--;

    cell_at(new_i).coords_x.push_back(x_coord_new);
    cell_at(new_i).death_rates.push_back(d);

    cell_death_rate_at(new_i) += d;
    total_death_rate += d;

    cell_population_at(new_i)++;
    total_population++;

    for (int i = new_i - cull_x; i < new_i + cull_x + 1; i++) {
        if (!periodic && (i < 0 || i >= cell_count_x))
            continue;
        for (int k = 0; k < cell_population_at(i); k++) {
            if (i == new_i && k == cell_population_at(new_i) - 1)
                continue;

            double distance;
            if (periodic) {
                if (i < 0) {
                    distance = abs(cell_at(new_i).coords_x.back() - cell_at(i).coords_x[k] + area_length_x);
                } else if (i >= cell_count_x) {
                    distance = abs(cell_at(new_i).coords_x.back() - cell_at(i).coords_x[k] - area_length_x);
                } else {
                    distance = abs(cell_at(new_i).coords_x.back() - cell_at(i).coords_x[k]);
                }
            } else {
                distance = abs(cell_at(new_i).coords_x.back() - cell_at(i).coords_x[k]);
            }

            if (distance > death_cutoff_r)
                continue;

            double interaction = dd * death_kernel(distance);

            cell_at(i).death_rates[k] += interaction;
            cell_at(new_i).death_rates.back() += interaction;

            cell_death_rate_at(i) += interaction;
            cell_death_rate_at(new_i) += interaction;

            total_death_rate += 2 * interaction;
        }
    }
}

void Grid1D::make_event() {
    if (total_population == 0)
        return;

    event_count++;
    std::exponential_distribution<> exp_dist(total_population * b + total_death_rate);
    time += exp_dist(rng);

    std::bernoulli_distribution bern_dist(total_population * b / (total_population * b + total_death_rate));
    if (!bern_dist(rng)) {
        kill_random();
    } else {
        spawn_random();
    }
}

void Grid1D::run_events(int events) {
    for (int i = 0; i < events; i++) {
        if (std::chrono::system_clock::now() > init_time + std::chrono::duration<double>(realtime_limit)) {
            realtime_limit_reached = true;
            return;
        }
        make_event();
    }
}

void Grid1D::run_for(double duration) {
    double start_time = time;
    while (time < start_time + duration) {
        if (std::chrono::system_clock::now() > init_time + std::chrono::duration<double>(realtime_limit)) {
            realtime_limit_reached = true;
            return;
        }
        make_event();
    }
}
