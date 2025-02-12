import sys

sys.path.append('/Users/haniftayarani/V2G_national/charging/src')
import time
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from temp import final_temp_adjustment
import process
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


# %%

def MakeItineraries(day_type="weekday"):
    # Start timer
    t0 = time.time()
    print('Loading NHTS data:', end='')

    # Load data
    trips = pd.read_csv('/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017/trippub.csv')
    temp = final_temp_adjustment()
    trips = pd.merge(trips, temp[["Energy_Consumption", "HHSTATE", "HHSTFIPS"]], on=["HHSTATE", "HHSTFIPS"], how="left")
    # Drop unnecessary columns
    trips = trips[['HOUSEID', 'PERSONID', 'TDTRPNUM', 'STRTTIME', 'ENDTIME', 'TRVLCMIN',
                   'TRPMILES', 'TRPTRANS', 'VEHID', 'WHYFROM', 'LOOP_TRIP', 'TRPHHVEH', "WHODROVE",
                   'PUBTRANS', 'TRIPPURP', 'DWELTIME', 'TDWKND', 'VMT_MILE', 'WHYTRP1S',
                   'TDCASEID', 'WHYTO', 'TRAVDAY', 'HOMEOWN', 'HHSIZE', 'HHVEHCNT',
                   'HHFAMINC', 'HHSTATE', 'HHSTFIPS', 'TDAYDATE', 'URBAN', 'URBANSIZE',
                   'URBRUR', 'GASPRICE', 'CENSUS_D', 'CENSUS_R', 'CDIVMSAR', 'VEHTYPE', "Energy_Consumption"]]

    # Filter trips based on day type
    if day_type == "weekday":
        trips = trips[trips["TDWKND"] == 2]  # Select only weekdays
    elif day_type == "weekend":
        trips = trips[trips["TDWKND"] == 1]  # Select only weekends
    elif day_type == "all":
        pass  # No filtering needed for all days
    else:
        raise ValueError("day_type must be one of 'weekday', 'weekend', or 'all'")

    # Further filtering of trips based on trip and vehicle criteria
    trip_veh = [3, 4, 5, 6]
    veh_type = [-1, 1, 2, 3, 4, 5]

    trips = trips[trips["TRPTRANS"].isin(trip_veh)]
    trips = trips[trips["TRPMILES"] > 0]
    trips = trips[trips["TRVLCMIN"] > 0]
    trips = trips[trips["VEHTYPE"].isin(veh_type)]
    trips = trips[trips["TRPMILES"] < 500]
    trips = trips[trips['PERSONID'] == trips['WHODROVE']]
    # Selecting for household vehicles
    trips = trips[(
        (trips['VEHID'] <= 2)
    )].copy()

    new_itineraries = trips
    print(' {:.4f} seconds'.format(time.time() - t0))

    t0 = time.time()
    print('Creating itinerary dicts:', end='')

    # Selecting for household vehicles
    new_itineraries = new_itineraries[(new_itineraries['TRPHHVEH'] == 1)].copy()

    # Get unique combinations of household, vehicle, and person IDs
    unique_combinations = new_itineraries[['HOUSEID', 'VEHID', 'PERSONID']].drop_duplicates()

    # Initialize an array to store each itinerary dictionary
    itineraries = np.array([None] * unique_combinations.shape[0])

    # Main loop: iterate over each unique household-vehicle-person combination in the test set
    for idx, row in tqdm(enumerate(unique_combinations.itertuples(index=False)), total=unique_combinations.shape[0]):
        hh_id = row.HOUSEID
        veh_id = row.VEHID
        person_id = row.PERSONID

        # Get trips for this specific household-vehicle-person combination
        trips_indices = np.argwhere(
            (new_itineraries['HOUSEID'] == hh_id) &
            (new_itineraries['VEHID'] == veh_id) &
            (new_itineraries['PERSONID'] == person_id)
        ).flatten()

        # Create the dictionary for the current household-vehicle-person combination
        itinerary_dict = {
            'trips': new_itineraries.iloc[trips_indices]
        }

        # Store in the array at the current index
        itineraries[idx] = itinerary_dict

    # Display time taken
    print('Itineraries creation took {:.4f} seconds'.format(time.time() - t0))

    # Set output file name based on day_type parameter
    if day_type == "weekday":
        output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekday.pkl'
    elif day_type == "weekend":
        output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekend.pkl'
    elif day_type == "all":
        output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_all_days.pkl'

    # Save itineraries to a pickle file
    t0 = time.time()
    print('Pickling outputs:', end='')
    pkl.dump(itineraries, open(output_file, 'wb'))
    print(' Done in {:.4f} seconds'.format(time.time() - t0))


def MakeItineraries_eVMT():
    codex_path = '/Users/haniftayarani/V2G_national/charging/codex.json'
    output_file = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_evmt.pkl'
    # Load input files and initial data
    data = process.load(codex_path)
    data_charging = data.get("charging", pd.DataFrame())
    data_trips = data.get("trips", pd.DataFrame())
    data_trips = data_trips.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    chevrolet_vehids = data_trips[data_trips['Make'].str.lower() == 'chevrolet']['vehicle_name'].unique().tolist()

    temp = final_temp_adjustment()

    # Define a function to extract battery capacity
    def get_battery_capacity(row):
        if 'Leaf' in row['vehicle_model']:
            return int(row['vehicle_model'].split()[1])
        elif 'Bolt' in row['vehicle_model']:
            return 66
        elif 'RAV4 EV' in row['vehicle_model']:
            return 41.8
        elif 'Model S' in row['Model']:
            # Extract capacity from the Model column if available
            model_parts = row['Model'].split()
            for part in model_parts:
                try:
                    return float(part.strip('P').strip('D'))
                except ValueError:
                    continue
            # If not found, use the vehicle_model and take the lower number
            if 'Model S' in row['Model']:
                # Replace underscores with spaces and extract numeric parts
                clean_model = row['vehicle_model'].replace('_', ' ')
                capacities = [float(s) for s in clean_model.split() if s.replace('.', '').isdigit()]
                if capacities:
                    return max(capacities)
        elif 'Model s' in row['Model']:
            # Extract capacity from the Model column if available
            model_parts = row['Model'].split()
            for part in model_parts:
                try:
                    return float(part.strip('P').strip('D'))
                except ValueError:
                    continue
            # If not found, use the vehicle_model and take the lower number
            if 'Model s' in row['Model']:
                # Replace underscores with spaces and extract numeric parts
                clean_model = row['vehicle_model'].replace('_', ' ')
                capacities = [float(s) for s in clean_model.split() if s.replace('.', '').isdigit()]
                if capacities:
                    return max(capacities)
        # Return NaN if no match
        return np.nan

    data_trips['battery_capacity_kwh'] = data_trips.apply(get_battery_capacity, axis=1)
    # Assign energy consumption values for California
    data_trips["Energy_Consumption"] = temp.loc[temp["HHSTATE"] == "CA", "Energy_Consumption"].iloc[0]
    data_trips["Energy_Consumption"] = data_trips["Energy_Consumption"] * 2236.94185
    # Function to convert datetime columns to HHMM format
    def convert_start_end_to_HHMM(df, start_column, end_column):
        def extract_HHMM(timestamp):
            try:
                ts = pd.Timestamp(timestamp)
                return int(f"{ts.hour}{ts.minute:02d}")
            except Exception:
                return None  # Handle invalid timestamps gracefully

        df["start_HHMM"] = df[start_column].apply(extract_HHMM)
        df["end_HHMM"] = df[end_column].apply(extract_HHMM)

        return df[["start_HHMM", "end_HHMM"]]

    # Rename columns for clarity and consistency
    data_trips.rename(columns={"Household": "HOUSEID", "duration": "TRVLCMIN", "distance": "TRPMILES", "vehicle_name": "VEHID", "battery_capacity_kwh": "BATTCAP"}, inplace=True)
    data_trips[["STRTTIME", "ENDTIME"]] = convert_start_end_to_HHMM(data_trips, "start_time_ (local)", "end_time_ (local)")
    data_trips.loc[data_trips["destination_label"] == "Home", "WHYTRP1S"] = 1
    data_trips.loc[data_trips["destination_label"] == "Work", "WHYTRP1S"] = 10
    data_trips.loc[((data_trips["destination_label"] != "Work") & (data_trips["destination_label"] != "Home")), "WHYTRP1S"] = 20
    data_trips["next_start_time"] = data_trips.groupby("VEHID")["start_time_ (local)"].shift(-1)
    data_trips["start_time_ (local)"] = pd.to_datetime(data_trips["start_time_ (local)"], errors='coerce')
    data_trips["next_start_time"] = pd.to_datetime(data_trips["next_start_time"], errors='coerce')
    data_trips["DWELTIME"] = ((data_trips["next_start_time"] - data_trips["start_time_ (local)"]).dt.total_seconds()) / 60
    data_trips.loc[data_trips["DWELTIME"] < 0, "DWELTIME"] = 6000
    data_trips.loc[data_trips["DWELTIME"] == 0, "DWELTIME"] = 1
    data_trips["TRVLCMIN"] = data_trips["TRVLCMIN"] / 60
    data_trips.loc[data_trips["TRVLCMIN"] < 1, "TRVLCMIN"] = 1
    data_trips.loc[data_trips["DWELTIME"].isna(), "DWELTIME"] = 6000
    data_trips.loc[data_trips["day_type"] == "weekday", "TDWKND"] = 2
    data_trips.loc[data_trips["day_type"] == "weekend", "TDWKND"] = 1
    data_trips["HHSTATE"] = "CA"
    data_trips["HHSTFIPS"] = 6
    data_trips.loc[data_trips["destination_label"] == "Home", "WHYTRP1S"] = 1
    data_trips.loc[data_trips["destination_label"] == "Work", "WHYTRP1S"] = 10
    data_trips.loc[data_trips["destination_label"] == "Other", "WHYTRP1S"] = 100
    data_trips = data_trips.loc[(data_trips["Model"] != "Leaf") & (data_trips["Model"] != "RAV4 EV")]
    # Prepare itineraries
    new_itineraries = data_trips
    new_itineraries = new_itineraries[["HOUSEID", "year", "month", "day", "STRTTIME", "ENDTIME", "VEHID", "TRVLCMIN", "TRPMILES", "DWELTIME", "WHYTRP1S", "TDWKND", "HHSTATE", "HHSTFIPS", "Energy_Consumption", "BATTCAP"]]
    new_itineraries.rename(columns={"year": "Year", "month": "Month", "day": "Day"}, inplace=True)
    new_itineraries.loc[(new_itineraries["TRPMILES"] > 120) & (new_itineraries['VEHID'].isin(chevrolet_vehids)), "TRPMILES"] = 120
    new_itineraries.loc[(new_itineraries["TRPMILES"] > 160) & (~new_itineraries['VEHID'].isin(chevrolet_vehids)), "TRPMILES"] = 160
    unique_combinations = new_itineraries[["HOUSEID", "VEHID"]].drop_duplicates()

    itineraries = np.array([None] * unique_combinations.shape[0])
    t0 = time.time()

    # Main loop: iterate over each unique household-vehicle-person combination
    for idx, row in tqdm(enumerate(unique_combinations.itertuples(index=False)), total=unique_combinations.shape[0]):
        hh_id = row.HOUSEID
        veh_id = row.VEHID
        # Get trips for this specific household-vehicle-person combination
        trips_indices = np.argwhere((new_itineraries['HOUSEID'] == hh_id) & (new_itineraries['VEHID'] == veh_id)).flatten()
        # Create the dictionary for the current household-vehicle-person combination
        itinerary_dict = {'trips': new_itineraries.iloc[trips_indices]}
        # Store in the array at the current index
        itineraries[idx] = itinerary_dict

    # Display time taken
    print('Itineraries creation took {:.4f} seconds'.format(time.time() - t0))

    # Save itineraries to file
    t0 = time.time()
    print('Pickling outputs:', end='')
    with open(output_file, 'wb') as f:
        pkl.dump(itineraries, f)
    print(' Done in {:.4f} seconds'.format(time.time() - t0))

# %%
# MakeItineraries(day_type="all")
# MakeItineraries_eVMT()

