import numpy as np
import os
import pandas as pd
import sys
import itertools
from scipy.stats import halfnorm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import simulation modules
from simulation import PyGrid1
from simulation.SDESimulator import SDESimulator

# Constants for output files and data directory
DATA_DIR = "data"
SIM_PARAMS_FILE = os.path.join(DATA_DIR, "sim_params.csv")
SDE_PHI_FILE = os.path.join(DATA_DIR, "sde_phi.csv")
IBM_COORDS_FILE = os.path.join(DATA_DIR, "ibm_coords.csv")


def generate_simulation_params(test=False):
    """
    Generate a DataFrame of simulation parameters for both IBM and SDE.

    In test mode, a small parameter set is generated; otherwise, a full sweep is created.

    Returns
    -------
    params_df : pandas.DataFrame
        DataFrame containing parameters for each simulation run.
    """
    L = 10.0          # Domain length
    N_sde = 1000      # Number of spatial grid points for SDE
    N_ibm = 100       # Number of cells for IBM
    dt = 0.01         # Time step for SDE
    T = 20.0          # Final simulation time
    lam = 2.0         # Initial density (individuals per unit length)
    base_seed = 42
    np.random.seed(base_seed)

    if test:
        parameter_sets = [{'b': 2.0, 'd': 1.0, 'dd': 0.1}]
        sigma_m_values = [0.1]
        sigma_w_values = [0.1]
        num_simulations = 2
    else:
        parameter_sets = [
            {'b': 2.0, 'd': 1.0, 'dd': 0.01},
            {'b': 1.0, 'd': 0.0, 'dd': 0.01}
        ]
        sigma_m_values = [0.1, 0.5, 1.0]
        sigma_w_values = [0.1, 0.5, 1.0]
        num_simulations = 100

    rows = []
    param_group_counter = 0
    for param_set in parameter_sets:
        b, d, dd = param_set['b'], param_set['d'], param_set['dd']
        for sigma_m, sigma_w in itertools.product(sigma_m_values, sigma_w_values):
            param_group_id = f"pg{param_group_counter}"
            param_group_counter += 1
            for sim_id in range(num_simulations):
                seed = np.random.randint(10 ** 9)
                global_id = f"{param_group_id}_{sim_id}"
                # SDE simulation row
                rows.append({
                    'global_id': global_id,
                    'simulation_id': sim_id,
                    'param_group': param_group_id,
                    'simulator': 'SDE',
                    'L': L,
                    'N': N_sde,
                    'dt': dt,
                    'T': T,
                    'b': b,
                    'd': d,
                    'dd': dd,
                    'lam': lam,
                    'sigma_m': sigma_m,
                    'sigma_w': sigma_w,
                    'seed': seed
                })
                # IBM simulation row
                rows.append({
                    'global_id': global_id,
                    'simulation_id': sim_id,
                    'param_group': param_group_id,
                    'simulator': 'IBM',
                    'L': L,
                    'N': N_ibm,
                    'T': T,
                    'b': b,
                    'd': d,
                    'dd': dd,
                    'lam': lam,
                    'sigma_m': sigma_m,
                    'sigma_w': sigma_w,
                    'seed': seed
                })

    return pd.DataFrame(rows)


def simulate_single_sde(sim_row):
    """
    Run a single SDE simulation using the provided parameters.
    
    Returns a list of dictionaries (one per grid point per time step) with columns:
    global_id, time_point, x_position, phi_value.
    """
    global_id = sim_row['global_id']
    seed = int(sim_row['seed'])
    L = sim_row['L']
    N = int(sim_row['N'])
    dt = sim_row['dt']
    T = sim_row['T']
    b = sim_row['b']
    d = sim_row['d']
    dd = sim_row['dd']
    lam = sim_row['lam']
    sigma_m = sim_row['sigma_m']
    sigma_w = sim_row['sigma_w']

    np.random.seed(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    # initial phi vector: uniform density lam
    phi_0 = np.ones(N) * lam

    output_rows = []
    # Record initial phi at time 0
    for x_val, phi_val in zip(x, phi_0):
        output_rows.append({
            'global_id': global_id,
            'time_point': 0,
            'x_position': x_val,
            'phi_value': phi_val
        })

    # Run simulation from time 1 to T
    for t_point in range(1, int(T) + 1):
        results = SDESimulator(
            L=L, N=N, dt=dt, T=t_point,
            b=b, d=d, dd=dd, lam=lam,
            phi_0=phi_0,
            sigma_m=sigma_m, sigma_w=sigma_w,
            seed=seed
        )
        phi_t = results['final_phi']
        for x_val, phi_val in zip(x, phi_t):
            output_rows.append({
                'global_id': global_id,
                'time_point': t_point,
                'x_position': x_val,
                'phi_value': phi_val
            })
        phi_0 = phi_t  # update for the next step

    return output_rows


def simulate_single_ibm(sim_row):
    """
    Run a single IBM simulation using the provided parameters.
    
    Returns a list of dictionaries (one per particle per time step) with columns:
    global_id, time_point, x_position.
    """
    global_id = sim_row['global_id']
    seed = int(sim_row['seed'])
    L = sim_row['L']
    N = int(sim_row['N'])
    T = sim_row['T']
    b = sim_row['b']
    d = sim_row['d']
    dd = sim_row['dd']
    lam = sim_row['lam']
    sigma_m = sim_row['sigma_m']
    sigma_w = sim_row['sigma_w']

    np.random.seed(seed)
    # Generate initial positions via a Poisson point process
    initial_positions = [[np.random.uniform(0, L)] for _ in range(np.random.poisson(lam * L))]
    if not initial_positions:
        # No particles generated; return an empty list.
        return []

    # Create kernels for the IBM simulation
    epsilon = 1e-3
    N_kernel = 1001
    uvals = np.linspace(0, 1 - epsilon, N_kernel)
    rvals = halfnorm.ppf(uvals, scale=sigma_m)
    birthX_1d = [uvals.tolist()]
    birthY_1d = [rvals.tolist()]

    def normal_1d_kernel(r, sigma=1.0):
        c = 1 / ((2 * np.pi * sigma ** 2) ** 0.5)
        return c * np.exp(-0.5 * (r ** 2) / (sigma ** 2))

    max_r = 5 * sigma_w
    distances = np.linspace(0, max_r, N_kernel)
    values = [normal_1d_kernel(r, sigma=sigma_w) for r in distances]
    deathX_1d = [[distances.tolist()]]
    deathY_1d = [[values]]
    cutoffs = [max_r]

    grid = PyGrid1(
        M=1,
        areaLen=[L],
        cellCount=[N],
        isPeriodic=True,
        birthRates=[b],
        deathRates=[d],
        ddMatrix=[dd],
        birthX=birthX_1d,
        birthY=birthY_1d,
        deathX_=deathX_1d,
        deathY_=deathY_1d,
        cutoffs=cutoffs,
        seed=seed,
        rtimeLimit=7200.0
    )

    grid.placePopulation([initial_positions])
    output_rows = []
    # Record initial positions at time 0
    positions = grid.get_all_particle_coords()[0]
    for pos in positions:
        output_rows.append({
            'global_id': global_id,
            'time_point': 0,
            'x_position': pos
        })

    # Run simulation for each time point from 1 to T
    for t_point in range(1, int(T) + 1):
        grid.run_for(1.0)
        positions = grid.get_all_particle_coords()[0]
        for pos in positions:
            output_rows.append({
                'global_id': global_id,
                'time_point': t_point,
                'x_position': pos
            })

    return output_rows


def append_rows_to_csv(filename, rows):
    """
    Append rows to a CSV file. If the file doesn't exist, write the header.
    """
    df = pd.DataFrame(rows)
    mode, header = ('a', False) if os.path.exists(filename) else ('w', True)
    with open(filename, mode, newline='') as f:
        df.to_csv(f, header=header, index=False)


def run_simulations(sim_params_file, sde_phi_file, ibm_coords_file):
    """
    Run all simulations (both SDE and IBM) in parallel.
    As each simulation completes, its results are immediately written to file.

    Parameters
    ----------
    sim_params_file : str
        CSV file with simulation parameters.
    sde_phi_file : str
        CSV file where SDE phi values will be saved.
    ibm_coords_file : str
        CSV file where IBM coordinates will be saved.
    """
    params_df = pd.read_csv(sim_params_file)
    futures = {}
    with ProcessPoolExecutor() as executor:
        # Submit all simulation tasks, differentiating by 'simulator' type.
        for _, row in params_df.iterrows():
            row_dict = row.to_dict()
            if row_dict['simulator'] == 'SDE':
                future = executor.submit(simulate_single_sde, row_dict)
                futures[future] = 'SDE'
            elif row_dict['simulator'] == 'IBM':
                future = executor.submit(simulate_single_ibm, row_dict)
                futures[future] = 'IBM'
        # Process tasks as they complete.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running simulations"):
            sim_type = futures[future]
            try:
                result_rows = future.result()
            except Exception as e:
                print(f"Error in {sim_type} simulation: {e}")
                continue

            if sim_type == 'SDE':
                append_rows_to_csv(sde_phi_file, result_rows)
            elif sim_type == 'IBM':
                append_rows_to_csv(ibm_coords_file, result_rows)


def prepare_data_directory():
    """Ensure the data directory exists and remove any central files to start fresh."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in [SIM_PARAMS_FILE, SDE_PHI_FILE, IBM_COORDS_FILE]:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run parameter sweep for SDE and IBM simulations')
    parser.add_argument('--test', action='store_true', help='Run a small test with fewer simulations')
    args = parser.parse_args()

    prepare_data_directory()

    # Step 1: Generate and save simulation parameters for both simulators.
    params_df = generate_simulation_params(test=args.test)
    params_df.to_csv(SIM_PARAMS_FILE, index=False)
    print("Simulation parameters saved.")

    # Step 2: Run all simulations in parallel.
    run_simulations(SIM_PARAMS_FILE, SDE_PHI_FILE, IBM_COORDS_FILE)
    print("All simulations complete.")
