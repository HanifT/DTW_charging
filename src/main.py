import sys
import os
# Add the src directory to the system path
sys.path.append(os.path.abspath("src"))
import matplotlib.pyplot as plt
import seaborn as sns
from load_pickle import load_charging_data
import pandas as pd
import numpy as np
from utilities import (compute_energy_added, split_charging_sessions, shift_time_to_zero_fixed, fit_exponential_to_charging,plot_fitted_logistic,plot_fitted_logarithmic,
                       plot_fitted_exponential, dtw_distance, trim_charging_sessions, fit_logistic_to_charging, fit_logarithmic_to_charging)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import random
# %%
df_fast, df_slow = load_charging_data()
# Apply to both df_fast and df_slow
df_fast = compute_energy_added(df_fast)
# df_slow = compute_energy_added(df_slow)

# Plot for Fast Chargers
# plot_energy_by_time_intervals(df_fast.iloc[0:1000].copy(), title="Fast Charging: Energy Added Over Time")
# # Plot for Slow Chargers
# plot_energy_by_time_intervals(df_slow.iloc[0:200].copy(), title="Slow Charging: Energy Added Over Time")
# %%

# Example usage with df_fast
df_fast_below_5, df_fast_above_5 = split_charging_sessions(df_fast)

# Apply function to df_fast
df_fast_shifted_fixed = shift_time_to_zero_fixed(df_fast_below_5)

# Example usage with df_fast_shifted
# plot_shifted_charging_events(df_fast_shifted_fixed, num_events=1000, y_column="energy_added")
# plot_shifted_charging_events(df_fast_shifted_fixed, num_events=1000, y_column="charge_energy_added")
# %%

# Trim data before fitting, keeping only up to 99% of final charge
df_trimmed = trim_charging_sessions(df_fast_shifted_fixed, y_column="charge_energy_added", cutoff_percent=95)


# Fit the equation for 'charge_energy_added' column
fit_results1 = fit_exponential_to_charging(df_trimmed, y_column="charge_energy_added")

fit_results2 = fit_logistic_to_charging(df_trimmed, y_column="charge_energy_added")

fit_results3 = fit_logarithmic_to_charging(df_trimmed, y_column="charge_energy_added")

# Example usage for charge_energy_added
plot_fitted_exponential(df_trimmed, fit_results1, y_column="charge_energy_added")

plot_fitted_logistic(df_trimmed, fit_results2, y_column="charge_energy_added")

plot_fitted_logarithmic(df_trimmed, fit_results3, y_column="charge_energy_added")

# %%

# Initialize lists to store time series and charge session IDs
charge_series = []
charge_ids = []

# Group by charge session
for charge_id, group in df_fast.groupby("charge_id"):
    # Extract the time series for energy_added
    ts = group[["energy_added"]].values  # Convert to NumPy array (shape: n_timesteps × 1)

    # Append time series and charge_id
    charge_series.append(ts)
    charge_ids.append(charge_id)

# Convert to numpy array (handling variable lengths)
charge_series = np.array(charge_series, dtype=object)

print(f" Extracted {len(charge_series)} charging sessions.")
print(f"Example shape of a single session: {charge_series[0].shape}")
#
# # Resample all time series to have 100 time steps
# charge_series_resampled = resample_time_series(charge_series, target_length=100)
#
# # Normalize after resampling
# scaler = TimeSeriesScalerMeanVariance()
# charge_series_scaled = scaler.fit_transform(charge_series_resampled)
# print(f" Shape after resampling and scaling: {charge_series_scaled.shape}")  # (n_samples, 100, 1)
#
# # Compute DTW distance matrix using FastDTW
# dtw_matrix_fast = fast_dtw_matrix(charge_series_scaled)
# print(f" Approximate DTW matrix computed with shape: {dtw_matrix_fast.shape}")


# %% Example

# Step 2: Determine number of chunks (each chunk contains 500 series)
chunk_size = 500
num_chunks = len(charge_series) // chunk_size  # Number of full chunks
if len(charge_series) % chunk_size != 0:
    num_chunks += 1  # Add one more chunk if there's a remainder

print(f"Total Series: {len(charge_series)}, Chunk Size: {chunk_size}, Number of Runs: {num_chunks}")

# Step 3: Process each chunk separately
all_cluster_results = []  # Store cluster results
print(f"Process starts:")
start_time_outer = time.time()

for chunk_idx in range(num_chunks):
    start_time = time.time()

    # Select the subset for this chunk
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(charge_series))  # Avoid out-of-bounds
    selected_series = charge_series[start_idx:end_idx]
    selected_ids = charge_ids[start_idx:end_idx]

    num_series = len(selected_series)
    dtw_distance_matrix = np.zeros((num_series, num_series))

    # Compute pairwise DTW distances
    for i in range(num_series):
        for j in range(i+1, num_series):  # symmetric matrix
            dist = dtw_distance(selected_series[i], selected_series[j])
            dtw_distance_matrix[i, j] = dist
            dtw_distance_matrix[j, i] = dist  # symmetry

    print(f"Chunk {chunk_idx+1}/{num_chunks}: Pairwise DTW distances calculated successfully.")

    # Convert square distance matrix to condensed form
    condensed_dist = squareform(dtw_distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    clusters = fcluster(linkage_matrix, t=5, criterion='maxclust')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Chunk {chunk_idx+1}/{num_chunks} Execution took {elapsed_time:.4f} seconds.")

    # Store results in DataFrame
    cluster_results = pd.DataFrame({"charge_id": selected_ids, "cluster": clusters, "Chunk": chunk_idx})
    all_cluster_results.append(cluster_results)

    # Save each chunk separately (Optional: Save files)
    cluster_results.to_csv(f"cluster_results_chunk_{chunk_idx+1}.csv", index=False)
    np.save(f"series_chunk_{chunk_idx+1}.npy", selected_series)  # Save the series

# Combine all chunks (Optional: Save final combined results)
final_cluster_results = pd.concat(all_cluster_results, ignore_index=True)
final_cluster_results.to_csv("all_cluster_results.csv", index=False)

print("All clustering processes completed successfully!")
end_time_outer = time.time()
elapsed_time_outer = end_time_outer - start_time_outer
print(f"Process End:{elapsed_time_outer:.4f} seconds.")

# %%
clusters = pd.read_csv("all_cluster_results.csv")
df_fast_clean = df_fast.copy()
df_fast_clean = df_fast_clean[df_fast_clean["energy_added"] < 5]
df_fast_clean = df_fast_clean[["charge_id", "created_at", "battery_level", "charge_energy_added", "battery_range", "time_to_full_charge", "energy_added"]]
# Group by 'charge_id' and calculate required values

# Ensure 'created_at' is in datetime format
df_fast_clean["created_at"] = pd.to_datetime(df_fast_clean["created_at"])

# Group by 'charge_id' and calculate required values
charging_summary = df_fast_clean.groupby("charge_id").apply(
    lambda group: pd.Series({
        "charging_duration": (group["created_at"].max() - group["created_at"].min()).total_seconds() / 60,  # Duration in minutes
        "soc_start": group["battery_level"].iloc[0],  # First battery level (SOC start)
        "soc_end": group["battery_level"].iloc[-1],  # Last battery level (SOC end)
        "total_energy_added": group["charge_energy_added"].iloc[-1]  # Energy added in the last row
    })
).reset_index()


charging_summary = pd.merge(charging_summary, clusters, on="charge_id", how="left")
# %%
