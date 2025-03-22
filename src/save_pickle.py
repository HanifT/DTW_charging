import pandas as pd
import os
import json
import os

# Define the path to the Data folder

data_folder = "Data/charging_data"

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Read and concatenate all CSV files into a single DataFrame
df_list = [pd.read_csv(os.path.join(data_folder, file)) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
df = df.drop(columns="Unnamed: 0")

df_fast = df[(df["fast_charger_present"] == 1) | (df["fast_charger_type"] == "Tesla") | (df["charger_power"] > 19)]
df_slow = df[~df.index.isin(df_fast.index)]

# Save the combined DataFrame as a pickle file
pickle_file = "charging_data_fast.pkl"
df_fast.to_pickle(pickle_file)

pickle_file = "charging_data_slow.pkl"
df_slow.to_pickle(pickle_file)

# Load config file
config_file = "codex.json"

with open(config_file, "r") as f:
    config = json.load(f)

# Get the pickle file path from config
pickle_file = config.get("pickle_file", "Data/charging_data.pkl")

# Load the pickle file
if os.path.exists(pickle_file):
    df = pd.read_pickle(pickle_file)
    print(f"Loaded {len(df)} rows from {pickle_file}.")
else:
    print(f"Pickle file not found at {pickle_file}. Please generate it first.")