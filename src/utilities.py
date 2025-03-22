import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import curve_fit
from fastdtw import fastdtw
from scipy.spatial.distance import squareform, pdist
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import seaborn as sns

def compute_energy_added(df, charge_id_col="charge_id", energy_col="charge_energy_added", time_col="created_at"):

    # Ensure the DataFrame is sorted by charge session and timestamp
    df = df.sort_values(by=[charge_id_col, time_col])

    # Compute the energy added per timestep within each charge session
    df["energy_added"] = df.groupby(charge_id_col)[energy_col].diff()

    # Replace NaN values (first row of each session) with the first recorded energy value
    first_values = df.groupby(charge_id_col)[energy_col].transform("first")
    df["energy_added"] = df["energy_added"].fillna(first_values)
    df.loc[df["energy_added"] < 0, "energy_added"] = 0
    # Convert timestamp to datetime if it's not already
    df[time_col] = pd.to_datetime(df[time_col])

    # Extract only the time (HH:MM:SS) as a string
    df["time"] = df[time_col].dt.strftime("%H:%M")

    return df


def plot_energy_by_time_intervals(df, title="Energy Added Over Time (By Time Intervals)"):
    df = df.copy()  # Avoid modifying the original DataFrame

    # Ensure time is in datetime format and extract hour
    df["time"] = df["created_at"].dt.strftime("%H:%M")
    df["hour"] = df["created_at"].dt.hour  # returns integers from 0 to 23

    # Define subplot grid (3 rows × 8 columns)
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharey=True)
    fig.suptitle(title, fontsize=16)

    # Iterate over 2-hour intervals
    for i, start_hour in enumerate(range(0, 24, 2)):  # 0-2, 2-4, ..., 22-24
        row, col = divmod(i, 4)  # Convert index to row and column position

        # Filter data for this time interval
        df_subset = df[(df["hour"] >= start_hour) & (df["hour"] < start_hour + 2)]

        ax = axes[row, col]  # Get current subplot
        if not df_subset.empty:
            for charge_id, group in df_subset.groupby("charge_id"):
                ax.plot(group["time"], group["energy_added"], linestyle="-", label=f"ID {charge_id}")

        # Formatting
        ax.set_title(f"{start_hour:02d}:00 - {start_hour+2:02d}:00", fontsize=10)
        ax.set_xlabel("Time (HH:MM)")
        ax.set_ylabel("Energy (kWh)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)

    # Adjust layout for readability
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def resample_time_series(ts_list, target_length=100):
    """
    Resample each time series to a fixed length.
    Handles single-point time series by repeating the value.
    """
    resampled_series = []

    for ts in ts_list:
        old_length = len(ts)

        # Handle case where time series has only 1 data point
        if old_length == 1:
            new_ts = np.full((target_length, 1), ts[0, 0])  # Repeat the same value
        else:
            old_indices = np.linspace(0, 1, old_length)  # Original indices
            new_indices = np.linspace(0, 1, target_length)  # New indices

            # Perform linear interpolation
            interpolator = interp1d(old_indices, ts[:, 0], kind="linear", fill_value="extrapolate")
            new_ts = interpolator(new_indices).reshape(-1, 1)  # Reshape to (target_length, 1)

        resampled_series.append(new_ts)

    return np.array(resampled_series)

def fast_dtw_matrix(time_series):
    """
    Compute an approximate DTW distance matrix using fastdtw.
    """
    n = len(time_series)
    dtw_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):  # Only compute upper triangle
            distance, _ = fastdtw(time_series[i].flatten(), time_series[j].flatten())  # Flatten for 1D
            dtw_matrix[i, j] = distance
            dtw_matrix[j, i] = distance  # Mirror it since DTW is symmetric

    return dtw_matrix


def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                          dtw_matrix[i, j-1],
                                          dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m]


def split_charging_sessions(df):
    # Ensure sorting by `charge_id` and `created_at`
    df_sorted = df.sort_values(by=['charge_id', 'created_at']).copy()

    # Get the first recorded `energy_added` for each `charge_id`
    first_energy = df_sorted.groupby('charge_id').first().reset_index()

    # Split based on `energy_added` threshold (5 kWh)
    df_below_5 = df_sorted[df_sorted['charge_id'].isin(first_energy[first_energy['energy_added'] < 5]['charge_id'])]
    df_above_5 = df_sorted[df_sorted['charge_id'].isin(first_energy[first_energy['energy_added'] >= 5]['charge_id'])]

    return df_below_5, df_above_5



def shift_time_to_zero_fixed(df):
    df = df.copy()  # Avoid modifying original DataFrame
    # Ensure sorting by 'charge_id' and 'created_at'
    df.sort_values(by=['charge_id', 'created_at'], inplace=True)
    # Convert 'time' to minutes since midnight
    df['time_minutes'] = df['time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    # Adjust for sessions that pass midnight
    def adjust_for_midnight(group):
        time_diffs = group['time_minutes'].diff().fillna(0)
        cumulative_shift = (time_diffs < 0).cumsum() * 1440  # 1440 minutes in a day
        group['time_minutes_adjusted'] = group['time_minutes'] + cumulative_shift
        return group

    df = df.groupby('charge_id', group_keys=False).apply(adjust_for_midnight)
    # Get the minimum time (start time) for each charge session
    min_time_per_charge = df.groupby('charge_id')['time_minutes_adjusted'].transform('min')
    # Shift time so that the session starts from 00:00
    df['shifted_time_minutes'] = df['time_minutes_adjusted'] - min_time_per_charge
    # Convert minutes back to HH:MM format
    df['shifted_time'] = df['shifted_time_minutes'].apply(lambda x: f"{x // 60:02d}:{x % 60:02d}")
    # Drop unnecessary columns
    df.drop(columns=['time_minutes', 'time_minutes_adjusted', 'shifted_time_minutes'], inplace=True)

    return df


def plot_shifted_charging_events(df, num_events=5, y_column="energy_added"):


    df = df.copy()  # Avoid modifying the original DataFrame

    # Convert 'shifted_time' to minutes for sorting and plotting
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # Select a subset of charge sessions to plot
    selected_charge_ids = df['charge_id'].unique()[:num_events]

    # Plot each selected charge session
    plt.figure(figsize=(12, 6))
    for charge_id in selected_charge_ids:
        session_data = df[df['charge_id'] == charge_id].sort_values(by='shifted_time_minutes')
        plt.plot(session_data['shifted_time_minutes'], session_data[y_column], linestyle='-', label=f"Charge {charge_id}")

    # Formatting the plot
    plt.xlabel("Shifted Time (minutes from session start)")
    plt.ylabel(y_column.replace("_", " ").title())  # Format y-axis label nicely
    plt.title(f"Charging Events ({y_column.replace('_', ' ').title()}) Over Shifted Time")
    # plt.legend()
    plt.grid(True)
    plt.show()


def plot_fitted_exponential(df, fit_results, y_column="energy_added"):

    df = df.copy()
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # Select a few charge sessions to visualize
    selected_charge_ids = fit_results["charge_id"].unique()[:100]  # Plot 5 sessions

    plt.figure(figsize=(12, 6))
    for charge_id in selected_charge_ids:
        # Get actual data
        session_data = df[df['charge_id'] == charge_id].sort_values(by="shifted_time_minutes")
        x_data = session_data['shifted_time_minutes'].values
        y_data = session_data[y_column].values

        # Get fitted parameters
        params = fit_results[fit_results['charge_id'] == charge_id]
        if params.empty:
            continue

        A, C = params.iloc[0][["A", "C"]]

        # Generate fitted curve
        x_fit = np.linspace(0, max(x_data), 100)
        y_fit = A - A * np.exp(-C * x_fit)

        # Plot actual vs fitted
        plt.scatter(x_data, y_data, label=f"Actual: {charge_id}", alpha=0.6)
        plt.plot(x_fit, y_fit, linestyle='--', label=f"Fitted: {charge_id}")

    # Formatting
    plt.xlabel("Shifted Time (minutes from session start)")
    plt.ylabel(y_column.replace("_", " ").title())
    plt.title(f"Charging Events with Fitted Exponential Curves ({y_column.replace('_', ' ').title()})")
    # plt.legend()
    plt.grid(True)
    plt.show()



def trim_charging_sessions(df, y_column="charge_energy_added", cutoff_percent=99):

    df = df.copy()
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    trimmed_dfs = []

    for charge_id, group in df.groupby("charge_id"):
        group = group.sort_values(by="shifted_time_minutes")

        if len(group) < 3:  # Ignore short sessions
            trimmed_dfs.append(group)
            continue

        final_value = group[y_column].iloc[-1]  # Last recorded charge value
        cutoff_value = (cutoff_percent / 100.0) * final_value  # Calculate threshold

        # Find first row where charge energy added reaches the cutoff
        cutoff_index = group[group[y_column] >= cutoff_value].index.min()

        if pd.notna(cutoff_index):
            group = group.loc[:cutoff_index]  # Keep only data up to the cutoff

        trimmed_dfs.append(group)

    return pd.concat(trimmed_dfs, ignore_index=True)


def exponential_model(x, A, C):
    return A - A * np.exp(-C * x)


def logistic_model(x, A, C, B):
    return A / (1 + np.exp(-C * (x - B)))


def logarithmic_model(x, A, B, C):
    return A * np.log(x + B) + C


def fit_exponential_to_charging(df, y_column="energy_added"):
    df = df.copy()
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    results = []

    for charge_id, group in df.groupby("charge_id"):
        group = group.sort_values(by="shifted_time_minutes")

        x_data = group['shifted_time_minutes'].values  # Time in minutes
        y_data = group[y_column].values

        if len(x_data) < 3:  # Need at least 3 points for a reliable fit
            continue

        try:
            # Fit the model
            popt, _ = curve_fit(exponential_model, x_data, y_data, maxfev=5000)
            A, C = popt
            results.append({"charge_id": charge_id, "A": A, "C": C})
        except RuntimeError:
            continue  # Skip if fitting fails

    return pd.DataFrame(results)

def fit_logistic_to_charging(df, y_column="energy_added"):
    df = df.copy()
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    results = []

    for charge_id, group in df.groupby("charge_id"):
        group = group.sort_values(by="shifted_time_minutes")

        x_data = group['shifted_time_minutes'].values
        y_data = group[y_column].values

        if len(x_data) < 3:  # Need at least 3 points for a reliable fit
            continue

        # Initial guesses
        A_init = max(y_data)  # Max charge added
        C_init = 0.1  # Growth rate guess
        B_init = np.median(x_data)  # Midpoint of charge event

        try:
            popt, _ = curve_fit(logistic_model, x_data, y_data, p0=[A_init, C_init, B_init], maxfev=10000)
            A, C, B = popt
            results.append({"charge_id": charge_id, "A": A, "C": C, "B": B})
        except RuntimeError:
            continue  # Skip if fitting fails

    return pd.DataFrame(results)

def fit_logarithmic_to_charging(df, y_column="energy_added"):
    df = df.copy()
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    results = []

    for charge_id, group in df.groupby("charge_id"):
        group = group.sort_values(by="shifted_time_minutes")

        x_data = group['shifted_time_minutes'].values
        y_data = group[y_column].values

        if len(x_data) < 3:  # Need at least 3 points for a reliable fit
            continue

        # Initial guesses
        A_init = (max(y_data) - min(y_data)) / np.log(max(x_data) + 1)  # Scale factor
        B_init = 1  # Small shift to prevent log(0)
        C_init = min(y_data)  # Offset

        try:
            popt, _ = curve_fit(logarithmic_model, x_data, y_data, p0=[A_init, B_init, C_init], maxfev=10000)
            A, B, C = popt
            results.append({"charge_id": charge_id, "A": A, "B": B, "C": C})
        except RuntimeError:
            continue  # Skip if fitting fails

    return pd.DataFrame(results)


def plot_fitted_logistic(df, fit_results, y_column="energy_added"):


    df = df.copy()
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # Select a few charge sessions to visualize
    selected_charge_ids = fit_results["charge_id"].unique()[:100]

    plt.figure(figsize=(12, 6))

    for charge_id in selected_charge_ids:
        # Get actual data
        session_data = df[df['charge_id'] == charge_id].sort_values(by="shifted_time_minutes")
        x_data = session_data['shifted_time_minutes'].values
        y_data = session_data[y_column].values

        # Get fitted parameters
        params = fit_results[fit_results['charge_id'] == charge_id]
        if params.empty:
            continue

        A, C, B = params.iloc[0][["A", "C", "B"]]

        # Generate fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = A / (1 + np.exp(-C * (x_fit - B)))

        # Plot actual vs fitted
        plt.scatter(x_data, y_data, alpha=0.6, label=f"Actual: {charge_id}")
        plt.plot(x_fit, y_fit, linestyle='--', label=f"Fitted: {charge_id}")

    # Formatting
    plt.xlabel("Shifted Time (minutes from session start)")
    plt.ylabel(y_column.replace("_", " ").title())
    plt.title(f"Charging Events with Fitted Logistic Curves ({y_column.replace('_', ' ').title()})")
    plt.grid(True)
    plt.show()

def plot_fitted_logarithmic(df, fit_results, y_column="energy_added"):

    df = df.copy()
    df['shifted_time_minutes'] = df['shifted_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # Select charge sessions to visualize
    selected_charge_ids = fit_results["charge_id"].unique()[:100]

    plt.figure(figsize=(12, 6))

    for charge_id in selected_charge_ids:
        # Get actual data
        session_data = df[df['charge_id'] == charge_id].sort_values(by="shifted_time_minutes")
        x_data = session_data['shifted_time_minutes'].values
        y_data = session_data[y_column].values

        # Get fitted parameters
        params = fit_results[fit_results['charge_id'] == charge_id]
        if params.empty:
            continue

        A, B, C = params.iloc[0][["A", "B", "C"]]

        # Ensure B is valid to avoid log(0)
        B = max(B, 1e-5)

        # Generate fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = A * np.log(x_fit + B) + C

        # Filter out invalid values (NaN, Inf)
        valid_indices = np.isfinite(y_fit)
        x_fit = x_fit[valid_indices]
        y_fit = y_fit[valid_indices]

        # Plot actual vs fitted
        plt.scatter(x_data, y_data, alpha=0.6, label=f"Actual: {charge_id}")
        plt.plot(x_fit, y_fit, linestyle='--', label=f"Fitted: {charge_id}")

    # Formatting
    plt.xlabel("Shifted Time (minutes from session start)")
    plt.ylabel(y_column.replace("_", " ").title())
    plt.title(f"Charging Events with Fitted Logarithmic Curves ({y_column.replace('_', ' ').title()})")
    plt.grid(True)
    plt.show()



def plot_box_by_chunk_cluster(df, column_name):

    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the dataset. Please enter a valid column name.")
        return

    unique_chunks = df["Chunk"].unique()

    for chunk in unique_chunks:
        plt.figure(figsize=(10, 6))
        chunk_data = df[df["Chunk"] == chunk]

        # Create boxplot
        sns.boxplot(x="cluster", y=column_name, data=chunk_data)
        plt.title(f"Box Plot of {column_name} for Chunk {chunk}")
        plt.xlabel("Cluster")
        plt.ylabel(column_name)
        plt.xticks(rotation=45)
        plt.grid(True)

        # Show plot
        plt.show()