import pandas as pd
import json
import os

def load_charging_data(config_file="codex.json"):
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{config_file} not found. Please create it with the correct paths.")

    # Get pickle file paths
    pickle_fast = config.get("pickle_fast")
    pickle_slow = config.get("pickle_slow")

    if not pickle_fast or not pickle_slow:
        raise ValueError("Pickle file paths missing in codex.json")

    df_fast, df_slow = None, None

    # Load fast charging data
    if os.path.exists(pickle_fast):
        df_fast = pd.read_pickle(pickle_fast)
        print(f"✅ Loaded {len(df_fast)} fast charging rows from {pickle_fast}.")
    else:
        print(f"❌ Fast charging pickle file not found at {pickle_fast}.")

    # Load slow charging data
    if os.path.exists(pickle_slow):
        df_slow = pd.read_pickle(pickle_slow)
        print(f"✅ Loaded {len(df_slow)} slow charging rows from {pickle_slow}.")
    else:
        print(f"❌ Slow charging pickle file not found at {pickle_slow}.")

    return df_fast, df_slow  # Returns DataFrames

# Example usage
