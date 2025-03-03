import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Set up parameters and load data
# -----------------------------
sns.set(style="whitegrid")
max_time = 20  # maximum time to display on the x-axis

# Load CSV files (assumed to be in the current working directory)
params_df = pd.read_csv("data/sim_params.csv")
sde_df = pd.read_csv("data/sde_phi.csv")
ibm_df = pd.read_csv("data/ibm_coords.csv")

# -----------------------------
# 2. Process SDE and IBM outputs
# -----------------------------
# For SDE: compute average density (simple average of phi_value per global_id and time_point)
sde_avg = sde_df.groupby(["global_id", "time_point"])["phi_value"].mean().reset_index()
sde_avg.rename(columns={"phi_value": "sde_density"}, inplace=True)

# For IBM: compute density by counting individuals and dividing by area length L = 10
ibm_counts = ibm_df.groupby(["global_id", "time_point"]).size().reset_index(name="count")
ibm_counts["ibm_density"] = ibm_counts["count"] / 10.0

# Merge SDE and IBM data on global_id and time_point
merged = pd.merge(sde_avg, ibm_counts, on=["global_id", "time_point"], how="inner")
# (Original diff was ibm_density - sde_density, but we'll recompute a percentage later)

# -----------------------------
# 3. Merge with simulation parameters
# -----------------------------
# Use only the SDE rows from sim_params.csv to avoid duplicates and extract needed parameters.
params_df_sde = params_df[params_df["simulator"] == "SDE"][['global_id','b','d','dd','sigma_m','sigma_w']]
merged_full = pd.merge(merged, params_df_sde, on="global_id", how="left")

# Extract replication info from global_id (assuming format "pgX_Y")
merged_full["replication"] = merged_full["global_id"].apply(lambda x: x.split("_")[1])

# -----------------------------
# 4. Define final parameter sets and ordering for subplots
# -----------------------------
# Two parameter sets for b, d, dd corresponding to plot 1 and plot 2.
parameter_sets = [
    {'b': 2.0, 'd': 1.0, 'dd': 0.01},
    {'b': 1.0, 'd': 0.0, 'dd': 0.01}
]
# sigma_m and sigma_w values for grid ordering (each grid is 3x3)
sigma_m_values = [0.1, 0.5, 1.0]   # will be plotted so that top row is highest σ_m
sigma_w_values = [0.1, 0.5, 1.0]   # left-to-right ordering

# For plotting, we want the grid rows ordered by sigma_m descending (top: 1.0, bottom: 0.1)
sigma_m_plot_order = sorted(sigma_m_values, reverse=True)

# -----------------------------
# 5. Create and save plots for each parameter set
# -----------------------------
for idx, param_set in enumerate(parameter_sets, start=1):
    # Filter data for the current parameter set (based on b, d, dd)
    current_data = merged_full[
        (merged_full["b"] == param_set["b"]) & 
        (merged_full["d"] == param_set["d"]) & 
        (merged_full["dd"] == param_set["dd"])
    ].copy()
    
    # Create a 3x3 grid for each combination of sigma_m and sigma_w
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
    
    for i, sigma_m in enumerate(sigma_m_plot_order):  # row index (top to bottom)
        for j, sigma_w in enumerate(sigma_w_values):    # column index (left to right)
            ax = axes[i, j]
            # Filter for this combination and time between 0 and max_time
            sub_data = current_data[
                (np.isclose(current_data["sigma_m"], sigma_m)) &
                (np.isclose(current_data["sigma_w"], sigma_w)) &
                (current_data["time_point"].between(0, max_time))
            ].copy()
            # Only keep integer time points
            sub_data = sub_data[np.isclose(sub_data["time_point"] % 1, 0)]
            
            # For percentage ratio calculation, first compute mean IBM density for each time point
            group_mean_ibm = sub_data.groupby("time_point")["ibm_density"].mean().reset_index().rename(columns={"ibm_density": "mean_ibm"})
            # Merge to sub_data so each row gets the corresponding mean IBM density
            sub_data = pd.merge(sub_data, group_mean_ibm, on="time_point", how="left")
            # Compute percentage difference: (SDE - IBM) / mean(IBM) * 100
            sub_data["perc_diff"] = (sub_data["sde_density"] - sub_data["ibm_density"]) / sub_data["mean_ibm"] * 100
            
            # Plot each replication run with low opacity (alpha ~0.05)
            for rep in sub_data["replication"].unique():
                rep_data = sub_data[sub_data["replication"] == rep].sort_values("time_point")
                ax.plot(rep_data["time_point"], rep_data["perc_diff"], color="gray", alpha=0.05)
            
            # Compute pointwise mean and standard error for percentage difference across replications
            grouped = sub_data.groupby("time_point")["perc_diff"]
            mean_perc = grouped.mean()
            std_perc = grouped.std()
            n = grouped.count()
            ci = 1.96 * std_perc / np.sqrt(n)
            lower = mean_perc - ci
            upper = mean_perc + ci
            
            # Plot the 95% confidence band (light blue fill) and mean percentage difference line (blue)
            ax.fill_between(mean_perc.index, lower, upper, color="lightblue", label="95% CI")
            ax.plot(mean_perc.index, mean_perc, color="blue", lw=2, label="Mean % diff")
            # Thick red horizontal line at y = 0
            ax.axhline(0, color="red", linewidth=1)
            
            ax.set_xticks(np.arange(0, max_time + 1, 1))
            ax.set_xlim(0, max_time)
            ax.set_ylim(-50, 50)
            ax.set_xlabel("Time")
            ax.set_ylabel("((SDE - IBM) / mean(IBM)) %")
            ax.set_title(f"σ_m = {sigma_m}, σ_w = {sigma_w}", fontsize=10)
            ax.legend(fontsize=8)
    
    fig.suptitle(f"Difference Plots for Parameter Set: b = {param_set['b']}, d = {param_set['d']}, dd = {param_set['dd']}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the figure with high resolution (dpi=300)
    fig.savefig(f"diff_plots_b{param_set['b']}_d{param_set['d']}_dd{param_set['dd']}.png", dpi=300)
    plt.show()
