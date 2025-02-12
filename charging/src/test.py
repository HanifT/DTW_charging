# %%
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/Users/haniftayarani/V2G_national/charging/src')
import pandas as pd

# %% Input eVMT


def load_pkl_to_dataframe(file_path):
    try:
        # Load the pickle file
        df = pd.read_pickle(file_path)
        print("File loaded successfully!")
        return df
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def combine_trips_to_dataframe(array_of_dicts, key_column="trips"):

    combined_dataframes = []
    for i, item in enumerate(array_of_dicts):
        if isinstance(item, dict) and key_column in item:
            df = item[key_column].copy()
            df["Source_Index"] = i  # Add a column to track the source
            combined_dataframes.append(df)
        else:
            print(f"Skipping item at index {i} as it does not contain the key '{key_column}'.")

    # Combine all DataFrames into one
    combined_dataframe = pd.concat(combined_dataframes, ignore_index=True)
    return combined_dataframe


file_path1 = "/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_all_days.pkl"
file_path2 = "/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_evmt.pkl"
df1 = load_pkl_to_dataframe(file_path1)
df2 = load_pkl_to_dataframe(file_path2)

df1 = combine_trips_to_dataframe(df1, key_column="trips")
df2 = combine_trips_to_dataframe(df2, key_column="trips")
