#ifndef GRID1D_H
#define GRID1D_H

#include <vector>
#include <random>
#include <chrono>

struct Cell_1d {
    std::vector<double> coords_x;
    std::vector<double> death_rates;

    Cell_1d() {}
};

class Grid1D {
public:
    double area_length_x;
    int cell_count_x;

    double b, d, dd;
    int seed;

    std::vector<double> initial_population_x;
    std::vector<double> death_y;
    std::vector<double> death_x;
    double death_cutoff_r;
    int death_spline_nodes;
    double death_step;

    std::vector<double> birth_inverse_rcdf_y;
    std::vector<double> birth_inverse_rcdf_x;
    int birth_inverse_rcdf_nodes;
    double birth_inverse_rcdf_step;

    bool periodic;
    double realtime_limit;
    bool realtime_limit_reached;

    int total_population;
    double total_death_rate;
    int event_count;
    double time;

    std::mt19937 rng;
    std::chrono::system_clock::time_point init_time;

    std::vector<Cell_1d> cells;
    std::vector<double> cell_death_rates;
    std::vector<int> cell_population;
    int cull_x;

    Grid1D(double area_length_x_, int cell_count_x_,
           double b_, double d_, double dd_, int seed_,
           const std::vector<double>& initial_population,
           const std::vector<double>& death_values,
           double death_cutoff_r_,
           const std::vector<double>& birth_values,
           bool periodic_, double realtime_limit_);

    void Initialize_death_rates();
    void kill_random();
    void spawn_random();
    void make_event();
    void run_events(int events);
    void run_for(double duration);

    double linear_interpolate(const std::vector<double>& xgdat, const std::vector<double>& gdat, double x);
    double death_kernel(double at);
    double birth_inverse_rcdf_kernel(double at);

    Cell_1d& cell_at(int i);
    double& cell_death_rate_at(int i);
    int& cell_population_at(int i);

    std::vector<double> get_x_coords_at_cell(int i);
    std::vector<double> get_death_rates_at_cell(int i);
    std::vector<double> get_all_x_coords();
    std::vector<double> get_all_death_rates();
};

#endif // GRID1D_H