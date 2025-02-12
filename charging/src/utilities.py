import sys
import time
import signal
import numpy as np
from scipy.stats import t
import os
import pandas as pd
import pickle
from contextlib import contextmanager
from scipy.special import comb
import json
# import optimization
import optimization_new
import process
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import itertools
from datetime import datetime, timedelta
# %%


class TimeoutException(Exception): pass


@contextmanager
def TimeLimit(seconds):
    def SignalHandler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, SignalHandler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


continental_us_fips = ([1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50,
                        51, 53, 54, 55, 56])

us_state_fips = ([1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                  27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50,
                  51, 53, 54, 55, 56])

alaska_fips = 2
hawaii_fips = 15

continental_us_abb = (['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'ID', 'IL',
                       'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
                       'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                       'VT', 'VA', 'WA', 'WV', 'WI', 'WY'])

us_state_abb = (['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL',
                 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
                 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'])


def GiniCoefficient(x):
    total = 0

    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))

    return total / (len(x) ** 2 * np.mean(x))


def BinomialDistribution(n, r, p):
    return comb(n, r) * p ** r * (1 - p) ** (n - r)


def IsIterable(value):
    return hasattr(value, '__iter__')


def TopNIndices(array, n):
    return sorted(range(len(array)), key=lambda i: array[i])[-n:]


def BottomNIndices(array, n):
    return sorted(range(len(array)), key=lambda i: array[i])[:n]


def T_Test(x, y, alpha):
    x_n = len(x)
    y_n = len(y)
    x_mu = x.mean()
    y_mu = y.mean()
    x_sig = x.std()
    y_sig = y.std()
    x_se = x_sig / np.sqrt(x_n)
    y_se = y_sig / np.sqrt(y_n)
    x_y_se = np.sqrt(x_se ** 2 + y_se ** 2)
    T = (x_mu - y_mu) / x_y_se
    DF = x_n + y_n
    T0 = t.ppf(1 - alpha, DF)
    P = (1 - t.cdf(np.abs(T), DF)) * 2
    return (P <= alpha), T, P, T0, DF


def FullFact(levels):
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    level_repeat = 1
    range_repeat = np.prod(levels).astype(int)
    for i in range(n):
        range_repeat /= levels[i]
        range_repeat = range_repeat.astype(int)
        lvl = []
        for j in range(levels[i]):
            lvl += [j] * level_repeat
        rng = lvl * range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
    return H.astype(int)


def Pythagorean(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Function for calculating distances between lon/lat pairs
def Haversine(lon1, lat1, lon2, lat2):
    r = 6372800  # [m]
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r


def RMSE(x, y):
    return np.sqrt(((x - y) ** 2).sum() / len(x))


def CondPrint(message, disp=True):
    if disp:
        print(message)


# Custom progress bar
class ProgressBar():

    def __init__(self, iterable, bar_length=20, disp=True, freq=1):

        self.iterable = iterable
        self.total = len(iterable)
        self.bar_length = bar_length
        self.disp = disp
        self.freq = freq

        if self.disp:
            self.update = self.Update
        else:
            self.update = self.Update_Null

    def __iter__(self):

        return PBIterator(self)

    def Update_Null(self, current, rt):
        pass

    def Update(self, current, rt):

        percent = float(current) * 100 / self.total
        arrow = '-' * int(percent / 100 * self.bar_length - 1) + '>'
        spaces = ' ' * (self.bar_length - len(arrow))
        itps = current / rt
        projrem = (self.total - current) / itps

        info_string = ("\r\033[32m %s [%s%s] (%d/%d) %d%%, %.2f %s, %.2f %s, %.2f %s \033[0m        \r"
                       % ('Progress', arrow, spaces, current - 1, self.total, percent, itps, 'it/s', rt, 'seconds elapsed',
                          projrem, 'seconds remaining'))

        sys.stdout.write(info_string)
        sys.stdout.flush()


# Custom iterator for progress bar
class PBIterator():
    def __init__(self, ProgressBar):

        self.ProgressBar = ProgressBar
        self.index = 0
        self.rt = 0
        self.t0 = time.time()

    def __next__(self):

        if self.index < len(self.ProgressBar.iterable):

            self.index += 1
            self.rt = time.time() - self.t0

            if self.index % self.ProgressBar.freq == 0:
                self.ProgressBar.update(self.index, self.rt)

            return self.ProgressBar.iterable[self.index - 1]

        else:

            self.index += 1
            self.rt = time.time() - self.t0

            self.ProgressBar.update(self.index, self.rt)

            if self.ProgressBar.disp:
                print('\n')

            raise StopIteration


def load_itineraries(day_type="weekday"):
    # Choose the filename based on the day_type parameter
    if day_type == "weekday":
        filename = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekday.pkl'
    elif day_type == "weekend":
        filename = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_weekend.pkl'
    elif day_type == "all":
        filename = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_all_days.pkl'
    elif day_type == "eVMT":
        filename = '/Users/haniftayarani/V2G_national/charging/Data/Generated_Data/itineraries_evmt.pkl'
    else:
        raise ValueError("day_type must be one of 'weekday', 'weekend', or 'all'")

    # Check if the file exists and load it
    if os.path.isfile(filename):
        itineraries = pd.read_pickle(filename)
        print(f'Loaded itineraries from {day_type} file.')
        return itineraries
    else:
        print(f'File {filename} does not exist.')
        return None


def read_pkl(file_path):
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    # Check if the data is an array and display or convert it accordingly
    if isinstance(data, np.ndarray):
        # If the data is a single array
        print("Data contains a single array:")
        print(data)
        # Optionally, convert to DataFrame if it's a 2D array
        if data.ndim == 2:
            df = pd.DataFrame(data)
            print("\nConverted to DataFrame:")
            print(df.head())
    elif isinstance(data, list) and all(isinstance(i, np.ndarray) for i in data):
        # If the data is a list of arrays
        print("Data contains a list of arrays:")
        for idx, array in enumerate(data):
            print(f"\nArray {idx + 1}:")
            print(array)
            # Optionally, convert each array to DataFrame if it's 2D
            if array.ndim == 2:
                df = pd.DataFrame(array)
                print("\nConverted to DataFrame:")
                print(df.head())
    else:
        print("The pickle file contains data that is not an array or list of arrays.")
    combined_df = pd.concat([entry['trips'] for entry in data], ignore_index=True)
    return combined_df


def nhts_state_count():
    nhts17 = pd.read_csv("/Users/haniftayarani/V2G_national/charging/Data/NHTS_2017/trippub.csv")
    nhts17 = nhts17[["HOUSEID", "PERSONID", "VEHID", "HHSTATE", "URBRUR", "HHSTFIPS", "VMT_MILE", "WTTRDFIN", "WHODROVE"]]
    nhts17 = nhts17[nhts17['PERSONID'] == nhts17['WHODROVE']]
    nhts17_grouped = nhts17.groupby(["HOUSEID", "PERSONID", "VEHID", "HHSTATE", "URBRUR", "HHSTFIPS", "WTTRDFIN"])["VMT_MILE"].mean().reset_index(drop=False)
    nhts17_grouped = nhts17_grouped[(nhts17_grouped["VEHID"] < 3) & (nhts17_grouped["VEHID"] > 0)]
    nhts17_grouped = nhts17_grouped[nhts17_grouped["PERSONID"] < 3]
    nhts17_grouped = nhts17_grouped[nhts17_grouped["VMT_MILE"] > 0]
    nhts17_grouped_mean = nhts17_grouped.groupby(["HHSTATE", "HHSTFIPS", "URBRUR"]).apply(
        lambda x: pd.Series({"VMT_MILE_weighted_avg": (x["VMT_MILE"] * x["WTTRDFIN"]).sum() / x["WTTRDFIN"].sum()})).reset_index()
    state_abbrev_to_name = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",    "NV": "Nevada", "NH": "New Hampshire",
        "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
        "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
        "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
        "WY": "Wyoming", "DC": "District of Columbia"}
    # Map the state names using HHSTFIPS
    nhts17_grouped_mean["State"] = nhts17_grouped_mean["HHSTATE"].map(state_abbrev_to_name)
    nhts17_grouped_mean_all = nhts17_grouped_mean.groupby(["State", "HHSTATE", "HHSTFIPS"])["VMT_MILE_weighted_avg"].mean().reset_index(drop=False)
    nhts_grouped = nhts17.groupby(["HOUSEID", "PERSONID", "VEHID"]).head(n=1)
    nstate_counts = nhts_grouped["HHSTATE"].value_counts().reset_index(drop=False)
    return nstate_counts, state_abbrev_to_name


def generate_itineraries(day_types="all", tile=14, states=["WY"], state_counts=None, electricity_price_file="electricity_prices.json", state_name=None):
    # Load electricity price data from JSON
    with open(electricity_price_file, 'r') as f:
        electricity_prices = json.load(f)  # Nested dictionary with state and hourly prices
    itineraries_total = load_itineraries(day_type=day_types)

    # Filter itineraries to include only those with HHSTATE in the provided states
    def filter_itineraries_by_states(itineraries, states):
        filtered_itineraries = [
            {"trips": itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)].copy()}
            for itinerary in itineraries
            if not itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)].empty
        ]
        return filtered_itineraries

    itineraries = filter_itineraries_by_states(itineraries_total, states)

    # Fetch the number of itineraries from state_counts
    if state_counts is not None:
        num_itinerarie = state_counts[state_counts["HHSTATE"].isin(states)]["count"].sum()
    else:
        raise ValueError("state_counts DataFrame is required to determine the number of itineraries.")

    battery_capacities = [80 * 3.6e6, 55 * 3.6e6, 66 * 3.6e6, 65 * 3.6e6, 62 * 3.6e6,
                          75 * 3.6e6, 135 * 3.6e6, 60 * 3.6e6, 70 * 3.6e6, 80 * 3.6e6]
    probabilities = [0.42, 0.1, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.25]
    battery_rng = np.random.default_rng(123)

    # Define a function to select a battery capacity based on the given probabilities
    def get_random_battery_capacity():
        return battery_rng.choice(battery_capacities, p=probabilities)

    # Other parameters for the itineraries
    itinerary_kwargs = {
        "tiles": tile,
        'initial_soc': 1,
        'final_soc': 1,
        'home_charger_likelihood': 1,
        'work_charger_likelihood': 0.75,
        "consumption": 782.928,
        'destination_charger_likelihood': 0.1,
        'home_charger_power': 6.1e3,
        'work_charger_power': 6.1e3,
        'destination_charger_power': 100.1e3,
        'max_soc': 1,
        'min_soc': .10,
        'min_dwell_event_duration': 15 * 60,
        'max_ad_hoc_event_duration': 7.2e4,
        'min_ad_hoc_event_duration': 0 * 60,
        'payment_penalty': 60,
        'travel_penalty': 60,
    }

    # Initialize an empty list to collect the results
    all_tailed_itineraries = []

    # Solver and itinerary parameters
    solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc'}

    # Loop over each itinerary in the list
    for n in range(min(num_itinerarie, len(itineraries)) - 1):
        # Select a battery capacity based on the defined distribution
        selected_battery_capacity = get_random_battery_capacity()
        selected_energy_consumption = itineraries[n]["trips"]["Energy_Consumption"].mean()
        state = itineraries[n]["trips"]["HHSTATE"].iloc[0]

        # Get electricity price for the state (e.g., residential rate)
        state_abbreviation = itineraries[n]["trips"]["HHSTATE"].iloc[0]
        state_full_name = state_name.get(state_abbreviation, None)

        if state_full_name is None:
            raise ValueError(f"State abbreviation {state_abbreviation} not found in mapping.")

        # Get electricity price for the full state name (e.g., residential rate)
        rates = electricity_prices.get(state_full_name, {})
        residential_rate = [rates.get("Residential", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        commercial_rate = [rates.get("Commercial", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        other_rate = 0.51  # Fixed rate for "other"

        # Flatten electricity_costs into a list of hourly rates
        if isinstance(residential_rate, dict) and "rate" in residential_rate:
            residential_rate = [residential_rate["rate"][str(hour)] for hour in range(8760)]

        if isinstance(commercial_rate, dict) and "rate" in commercial_rate:
            commercial_rate = [commercial_rate["rate"][str(hour)] for hour in range(8760)]

        itinerary_kwargs['battery_capacity'] = selected_battery_capacity
        itinerary_kwargs['consumption'] = selected_energy_consumption
        itinerary_kwargs['residential_rate'] = residential_rate
        itinerary_kwargs['commercial_rate'] = commercial_rate
        itinerary_kwargs['other_rate'] = other_rate
        # Instantiate the EVCSP class for each itinerary
        problem = optimization_new.EVCSP(itineraries[n], itinerary_kwargs=itinerary_kwargs)
        problem.Solve(solver_kwargs)

        # Print status and SIC for tracking
        print(f'Itinerary {n}: solver status - {problem.solver_status}, termination condition - {problem.solver_termination_condition}')
        print(f'Itinerary {n}: SIC - {problem.sic}')

        # Repeat (tail) the itinerary across tiles
        # tailed_itinerary = pd.concat([itineraries[n]['trips']] * tile, ignore_index=True)
        original_columns = itineraries[n]['trips'].columns
        tailed_itinerary = pd.concat([itineraries[n]['trips'][original_columns]] * tile, ignore_index=True)

        # Add the SOC from the solution to the tailed itinerary
        soc_values = problem.solution['soc', 0].values  # Extract SOC values from problem.solution
        tailed_itinerary['SOC'] = soc_values[:len(tailed_itinerary)]

        # Add the SIC as a column to the tailed itinerary (repeated for each row)
        tailed_itinerary['SIC'] = problem.sic
        tailed_itinerary['Charging_cost'] = problem.charging_cost
        # Add the battery capacity as a column to the tailed itinerary
        tailed_itinerary['Battery Capacity'] = selected_battery_capacity  # <-- Add this line

        # Append the tailed itinerary with SOC and SIC to the results list
        all_tailed_itineraries.append(tailed_itinerary)

    # Concatenate all the individual DataFrames into one final DataFrame
    final_df = pd.concat(all_tailed_itineraries, ignore_index=True)
    final_df.name = f"final_df_{day_types}_{'_'.join(states)}"
    final_df['TripOrder'] = final_df.groupby('HOUSEID').cumcount() + 1
    return final_df

def generate_itineraries(day_types="all", tile=14, states=["WY"], state_counts=None, electricity_price_file="electricity_prices.json", state_name=None):
    # Load electricity price data from JSON
    with open(electricity_price_file, 'r') as f:
        electricity_prices = json.load(f)
    itineraries_total = load_itineraries(day_type=day_types)

    # Filter itineraries to include only those with HHSTATE in the provided states
    def filter_itineraries_by_states(itineraries, states):
        return [
            {"trips": itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)].copy()}
            for itinerary in itineraries
            if not itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)].empty
        ]

    itineraries = filter_itineraries_by_states(itineraries_total, states)

    # Fetch the number of itineraries from state_counts
    if state_counts is not None:
        num_itineraries = state_counts[state_counts["HHSTATE"].isin(states)]["count"].sum()
    else:
        raise ValueError("state_counts DataFrame is required to determine the number of itineraries.")

    # Battery assignment
    battery_capacities = [80 * 3.6e6, 55 * 3.6e6, 66 * 3.6e6, 65 * 3.6e6, 62 * 3.6e6,
                          75 * 3.6e6, 135 * 3.6e6, 60 * 3.6e6, 70 * 3.6e6, 80 * 3.6e6]
    probabilities = [0.42, 0.1, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.25]
    battery_rng = np.random.default_rng(123)

    def get_random_battery_capacity():
        return battery_rng.choice(battery_capacities, p=probabilities)

    solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc'}
    all_tailed_itineraries = []

    for n in range(min(num_itineraries, len(itineraries))):
        selected_battery_capacity = get_random_battery_capacity()
        selected_energy_consumption = itineraries[n]["trips"]["Energy_Consumption"].mean()
        state = itineraries[n]["trips"]["HHSTATE"].iloc[0]

        # Random start date
        start_date = datetime(2023, np.random.randint(1, 13), np.random.randint(1, 29))

        # Apply start date and increment for each tile
        trips = itineraries[n]['trips'].copy()
        tailed_itinerary = pd.concat([
            trips.assign(
                date=start_date + timedelta(days=i),
                Month=(start_date + timedelta(days=i)).month,
                Day=(start_date + timedelta(days=i)).day
            ) for i in range(tile)
        ], ignore_index=True)

        state_full_name = state_name.get(state, None)
        if state_full_name is None:
            raise ValueError(f"State abbreviation {state} not found in mapping.")

        rates = electricity_prices.get(state_full_name, {})
        residential_rate = [rates.get("Residential", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        commercial_rate = [rates.get("Commercial", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        other_rate = 0.51

        itinerary_kwargs = {
            "tiles": tile,
            'initial_soc': 1,
            'final_soc': 1,
            'home_charger_likelihood': 1,
            'work_charger_likelihood': 0.75,
            'destination_charger_likelihood': 0.1,
            'consumption': selected_energy_consumption,
            'home_charger_power': 6.1e3,
            'work_charger_power': 6.1e3,
            'destination_charger_power': 100.1e3,
            'max_soc': 1,
            'min_soc': .10,
            'min_dwell_event_duration': 15 * 60,
            'max_ad_hoc_event_duration': 7.2e4,
            'min_ad_hoc_event_duration': 0 * 60,
            'payment_penalty': 60,
            'travel_penalty': 60,
            'battery_capacity': selected_battery_capacity,
            'residential_rate': residential_rate,
            'commercial_rate': commercial_rate,
            'other_rate': other_rate
        }

        problem = optimization_new.EVCSP_delay({"trips": tailed_itinerary}, itinerary_kwargs=itinerary_kwargs)
        problem.Solve(solver_kwargs)

        tailed_itinerary['SOC'] = problem.solution['soc', 0].values[:len(tailed_itinerary)]
        tailed_itinerary['SIC'] = problem.sic
        tailed_itinerary['Charging_cost'] = problem.solution.get('charging_cost', np.nan)[:len(tailed_itinerary)]
        tailed_itinerary['Battery Capacity'] = selected_battery_capacity

        all_tailed_itineraries.append(tailed_itinerary)

    final_df = pd.concat(all_tailed_itineraries, ignore_index=True)
    final_df['TripOrder'] = final_df.groupby('HOUSEID').cumcount() + 1
    return final_df

def generate_itineraries_evmt_by_days(day_types="eVMT", tile=1, states=["CA"], days_to_select=100, electricity_price_file="/path/to/weighted_hourly_prices.json", state_name=None):
    # Load electricity price data from JSON
    with open(electricity_price_file, 'r') as f:
        electricity_prices = json.load(f)  # Nested dictionary with state and hourly prices

    itineraries_total = load_itineraries(day_type=day_types)

    # Filter itineraries to include only those with HHSTATE in the provided states
    def filter_itineraries_by_states(itineraries, states):
        excluded_indices = {}  # Exclude 69, 76, 78
        filtered_itineraries = []

        for idx, itinerary in enumerate(itineraries):
            if idx in excluded_indices:
                continue
            if not itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)].empty:
                filtered_itineraries.append({"trips": itinerary["trips"].copy()})

        return filtered_itineraries

    itineraries = filter_itineraries_by_states(itineraries_total, states)

    # Initialize a list for the filtered itineraries based on the number of days
    filtered_itineraries = []
    all_filtered_trips = []

    for itinerary in itineraries:
        # itinerary = itineraries[0]
        trips = itinerary["trips"]
        # test = itineraries[0]["trips"]
        # Convert 'YEAR', 'MONTH', 'DAY' columns to datetime format
        trips['date'] = pd.to_datetime(trips[['Year', 'Month', 'Day']])

        # Sort trips by vehicle and date to ensure correct chronological order
        trips = trips.sort_values(by=['HOUSEID', 'VEHID', 'date']).reset_index(drop=True)

        # Calculate the date range for this vehicle
        min_date = trips['date'].min()
        max_date = trips['date'].max()
        total_days = (max_date - min_date).days + 1

        # Determine the number of days to select
        if total_days <= days_to_select:
            selected_days = total_days  # Use all available days if fewer than requested
        else:
            selected_days = days_to_select

        # Filter trips to include only those within the selected number of days from min_date
        end_date = min_date + pd.Timedelta(days=selected_days - 1)
        filtered_trips = trips[trips['date'] <= end_date]

        # Re-sort filtered trips by vehicle and date after filtering
        filtered_trips = filtered_trips.sort_values(by=['HOUSEID', 'VEHID', 'date']).reset_index(drop=True)

        all_filtered_trips.append(filtered_trips)

        # Add the filtered trips to the list of itineraries
        filtered_itinerary = {"trips": filtered_trips}
        filtered_itineraries.append(filtered_itinerary)

    # Before concatenating, check if duplicates exist
    data_df_test = pd.concat(all_filtered_trips, ignore_index=True)

    # Solver and itinerary parameters
    solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc'}

    # Initialize an empty list to collect the results
    all_tailed_itineraries = []

    # Loop over each filtered itinerary and solve the problem
    for n, itinerary in enumerate(filtered_itineraries):
        # Select battery capacity and energy consumption for each itinerary
        selected_battery_capacity = itinerary["trips"]["BATTCAP"].iloc[0] * 3.6e6  # Convert from kWh to Joules
        selected_energy_consumption = itinerary["trips"]["Energy_Consumption"].mean()
        state = itinerary["trips"]["HHSTATE"].iloc[0]

        # Get electricity price for the state
        state_full_name = state_name.get(state, None)
        if state_full_name is None:
            raise ValueError(f"State abbreviation {state} not found in mapping.")

        # Retrieve electricity rates
        rates = electricity_prices.get(state_full_name, {})
        residential_rate = [rates.get("Residential", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        commercial_rate = [rates.get("Commercial", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        other_rate = 0.52  # Fixed rate for "other"

        # Set itinerary parameters for optimization
        itinerary_kwargs = {
            "tiles": tile,
            'initial_soc': 0.8,
            'final_soc': 0.8,
            'home_charger_likelihood': 0.4,
            'work_charger_likelihood': 0.4,
            'destination_charger_likelihood': 0.2,
            "consumption": selected_energy_consumption,
            'home_charger_power': 6.6*3600000,
            'work_charger_power': 7.2*3600000,
            'destination_charger_power': 150.1*3600000,
            "ad_hoc_charger_power": 150.1*3600000,
            'max_soc': 1,
            'min_soc': 0.2,
            'min_dwell_event_duration': 0.25,
            'max_ad_hoc_event_duration': 14.4e4,
            'min_ad_hoc_event_duration': 0,
            'payment_penalty': 60,
            'time_penalty': 60,
            'travel_penalty': 60,
            "ad_hoc_charge_time_penalty": 60,
            'battery_capacity': selected_battery_capacity,
            'residential_rate': residential_rate,
            'commercial_rate': commercial_rate,
            'other_rate': other_rate,
        }

        # Instantiate and solve the optimization problem
        problem = optimization_new.EVCSP(itinerary, itinerary_kwargs=itinerary_kwargs)
        problem.Solve(solver_kwargs)

        # Print solver status and SIC for tracking
        print(f'Itinerary {n}: HHID - {itinerary["trips"]["HOUSEID"].iloc[0]}')
        print(f'Itinerary {n}: solver status - {problem.solver_status}, termination condition - {problem.solver_termination_condition}')
        print(f'Itinerary {n}: SIC - {problem.sic}')

        # Repeat (tail) the itinerary across tiles and append SOC and SIC results
        original_columns = itinerary["trips"].columns
        tailed_itinerary = itinerary["trips"][original_columns].copy()

        # Add the SOC and SIC values to the itinerary
        soc_values = problem.solution['soc'].values
        tailed_itinerary['SOC'] = soc_values[:len(tailed_itinerary)]
        charging_start_key = ('charging_start_time_HHMM', 0)
        charging_end_key = ('charging_end_time_HHMM', 0)
        charging_distribution_key = ('hourly_charging_details', 0)
        num_trips = len(tailed_itinerary)

        # Assign SOC before and after the trip
        tailed_itinerary['itineraries'] = n
        tailed_itinerary['SOC_Trip_start'] = problem.solution.get('soc_start_trip', np.nan)[:num_trips]
        tailed_itinerary['SOC_Trip_end'] = problem.solution.get('soc_end_trip', np.nan)[:num_trips]
        tailed_itinerary['SIC'] = problem.sic
        tailed_itinerary['Charging_cost'] = problem.solution.get('charging_cost', np.nan)[:num_trips]
        tailed_itinerary['Charging_kwh'] = problem.solution.get('charging_kwh_total', np.nan)[:num_trips]
        tailed_itinerary['Battery Capacity'] = selected_battery_capacity

        # Safely get the charging start times
        charging_starts = problem.solution.get(charging_start_key, [np.nan] * num_trips)[:num_trips]

        # Safely get the charging end times
        charging_ends = problem.solution.get(charging_end_key, [np.nan] * num_trips)[:num_trips]

        # Safely get the charging kWh distribution (hourly)
        charging_kwh_distribution = problem.solution.get(charging_distribution_key, [np.nan] * num_trips)[:num_trips]

        # Store them in the tailed_itinerary DataFrame
        tailed_itinerary['Charging_Start_Time'] = charging_starts
        tailed_itinerary['Charging_End_Time'] = charging_ends
        tailed_itinerary['Charging_kwh_distribution'] = charging_kwh_distribution

        # Append the tailed itinerary to the results list
        all_tailed_itineraries.append(tailed_itinerary)

        # Initialize the new columns with NaN
        tailed_itinerary['SOC_charging_start'] = np.nan
        tailed_itinerary['SOC_charging_end'] = np.nan

        # Build a mask for rows where charging occurs (start time is not NaN)
        mask_charging = ~tailed_itinerary['Charging_Start_Time'].isna()

        # 4) For rows with charging:
        #    - SOC_charging_end = SOC_Trip_end (final SOC after the charge)
        #    - SOC_charging_start = (SOC_Trip_start - trip fraction),
        #      i.e., battery fraction immediately after the trip but before charging.

        tailed_itinerary.loc[mask_charging, 'SOC_charging_end'] = \
            tailed_itinerary.loc[mask_charging, 'SOC']

        tailed_itinerary.loc[mask_charging, 'SOC_charging_start'] = \
            tailed_itinerary.loc[mask_charging, 'SOC_Trip_end']

        tailed_itinerary.loc[mask_charging, 'SOC_Trip_end'] = \
            tailed_itinerary.loc[mask_charging, 'SOC_Trip_end']

    # Concatenate all the individual DataFrames into one final DataFrame
    final_df = pd.concat(all_tailed_itineraries, ignore_index=True)
    final_df.name = f"final_df_{day_types}_{'_'.join(states)}"
    final_df["PERSONID"] = 1
    final_df['TripOrder'] = final_df.groupby('HOUSEID').cumcount() + 1
    final_df.drop(["date", "SOC", "Battery Capacity", "PERSONID"], axis=1, inplace=True)
    return final_df


def generate_itineraries_evmt_by_days_delay(day_types="eVMT", tile=1, states=["CA"], days_to_select=100, electricity_price_file="/path/to/weighted_hourly_prices.json", state_name=None):
    # Load electricity price data from JSON
    with open(electricity_price_file, 'r') as f:
        electricity_prices = json.load(f)  # Nested dictionary with state and hourly prices

    itineraries_total = load_itineraries(day_type=day_types)

    # Filter itineraries to include only those with HHSTATE in the provided states
    def filter_itineraries_by_states(itineraries, states):
        excluded_indices = {}  # Exclude 69, 76, 78
        filtered_itineraries = []

        for idx, itinerary in enumerate(itineraries):
            if idx in excluded_indices:
                continue
            if not itinerary["trips"][itinerary["trips"]["HHSTATE"].isin(states)].empty:
                filtered_itineraries.append({"trips": itinerary["trips"].copy()})

        return filtered_itineraries

    itineraries = filter_itineraries_by_states(itineraries_total, states)

    # Initialize a list for the filtered itineraries based on the number of days
    filtered_itineraries = []
    all_filtered_trips = []

    for itinerary in itineraries:
        # itinerary = itineraries[0]
        trips = itinerary["trips"]
        # test = itineraries[0]["trips"]
        # Convert 'YEAR', 'MONTH', 'DAY' columns to datetime format
        trips['date'] = pd.to_datetime(trips[['Year', 'Month', 'Day']])

        # Sort trips by vehicle and date to ensure correct chronological order
        trips = trips.sort_values(by=['HOUSEID', 'VEHID', 'date']).reset_index(drop=True)

        # Calculate the date range for this vehicle
        min_date = trips['date'].min()
        max_date = trips['date'].max()
        total_days = (max_date - min_date).days + 1

        # Determine the number of days to select
        if total_days <= days_to_select:
            selected_days = total_days  # Use all available days if fewer than requested
        else:
            selected_days = days_to_select

        # Filter trips to include only those within the selected number of days from min_date
        end_date = min_date + pd.Timedelta(days=selected_days - 1)
        filtered_trips = trips[trips['date'] <= end_date]

        # Re-sort filtered trips by vehicle and date after filtering
        filtered_trips = filtered_trips.sort_values(by=['HOUSEID', 'VEHID', 'date']).reset_index(drop=True)

        all_filtered_trips.append(filtered_trips)

        # Add the filtered trips to the list of itineraries
        filtered_itinerary = {"trips": filtered_trips}
        filtered_itineraries.append(filtered_itinerary)

    # Before concatenating, check if duplicates exist
    data_df_test = pd.concat(all_filtered_trips, ignore_index=True)

    # Solver and itinerary parameters
    solver_kwargs = {'_name': 'cbc', 'executable': '/opt/homebrew/bin/cbc'}

    # Initialize an empty list to collect the results
    all_tailed_itineraries = []

    # Loop over each filtered itinerary and solve the problem
    for n, itinerary in enumerate(filtered_itineraries):
        # Select battery capacity and energy consumption for each itinerary
        selected_battery_capacity = itinerary["trips"]["BATTCAP"].iloc[0] * 3.6e6  # Convert from kWh to Joules
        selected_energy_consumption = itinerary["trips"]["Energy_Consumption"].mean()
        state = itinerary["trips"]["HHSTATE"].iloc[0]

        # Get electricity price for the state
        state_full_name = state_name.get(state, None)
        if state_full_name is None:
            raise ValueError(f"State abbreviation {state} not found in mapping.")

        # Retrieve electricity rates
        rates = electricity_prices.get(state_full_name, {})
        residential_rate = [rates.get("Residential", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        commercial_rate = [rates.get("Commercial", {}).get("rate", {}).get(str(hour), 0) for hour in range(8760)]
        other_rate = 0.52  # Fixed rate for "other"

        # Set itinerary parameters for optimization
        itinerary_kwargs = {
            "tiles": tile,
            'initial_soc': 0.8,
            'final_soc': 0.8,
            'home_charger_likelihood': 1,
            'work_charger_likelihood': 0.8,
            'destination_charger_likelihood': 0.2,
            'midnight_charging_prob': 0.5,
            "consumption": selected_energy_consumption,
            'home_charger_power': 6.6 * 3600000,
            'work_charger_power': 7.2 * 3600000,
            'destination_charger_power': 150.1 * 3600000,
            "ad_hoc_charger_power": 150.1 * 3600000,
            'max_soc': 1,
            'min_soc': 0.05,
            'min_dwell_event_duration': 0.25,
            'max_ad_hoc_event_duration': 14.4e4,
            'min_ad_hoc_event_duration': 0,
            'payment_penalty': 60,
            'travel_penalty': 60,
            "ad_hoc_charge_time_penalty": 60,
            'battery_capacity': selected_battery_capacity,
            'residential_rate': residential_rate,
            'commercial_rate': commercial_rate,
            'other_rate': other_rate,
        }

        # Instantiate and solve the optimization problem
        problem = optimization_new.EVCSP_delay(itinerary, itinerary_kwargs=itinerary_kwargs)
        problem.Solve(solver_kwargs)

        # Print solver status and SIC for tracking
        print(f'Itinerary {n}: HHID - {itinerary["trips"]["HOUSEID"].iloc[0]}')
        print(f'Itinerary {n}: solver status - {problem.solver_status}, termination condition - {problem.solver_termination_condition}')
        print(f'Itinerary {n}: SIC - {problem.sic}')

        # Repeat (tail) the itinerary across tiles and append SOC and SIC results
        original_columns = itinerary["trips"].columns
        tailed_itinerary = itinerary["trips"][original_columns].copy()

        # Add the SOC and SIC values to the itinerary
        soc_values = problem.solution['soc'].values
        tailed_itinerary['SOC'] = soc_values[:len(tailed_itinerary)]
        charging_start_key = ('charging_start_time_HHMM', 0)
        charging_end_key = ('charging_end_time_HHMM', 0)
        charging_distribution_key = ('hourly_charging_details', 0)
        num_trips = len(tailed_itinerary)

        # Assign SOC before and after the trip
        tailed_itinerary['itineraries'] = n
        tailed_itinerary['SOC_Trip_start'] = problem.solution.get('soc_start_trip', np.nan)[:num_trips]
        tailed_itinerary['SOC_Trip_end'] = problem.solution.get('soc_end_trip', np.nan)[:num_trips]
        tailed_itinerary['SIC'] = problem.sic
        tailed_itinerary['Charging_cost'] = problem.solution.get('charging_cost', np.nan)[:num_trips]
        tailed_itinerary['Charging_kwh'] = problem.solution.get('charging_kwh_total', np.nan)[:num_trips]
        tailed_itinerary['Battery Capacity'] = selected_battery_capacity

        # Safely get the charging start times
        charging_starts = problem.solution.get(charging_start_key, [np.nan] * num_trips)[:num_trips]

        # Safely get the charging end times
        charging_ends = problem.solution.get(charging_end_key, [np.nan] * num_trips)[:num_trips]

        # Safely get the charging kWh distribution (hourly)
        charging_kwh_distribution = problem.solution.get(charging_distribution_key, [np.nan] * num_trips)[:num_trips]

        # Store them in the tailed_itinerary DataFrame
        tailed_itinerary['Charging_Start_Time'] = charging_starts
        tailed_itinerary['Charging_End_Time'] = charging_ends
        tailed_itinerary['Charging_kwh_distribution'] = charging_kwh_distribution

        # Append the tailed itinerary to the results list
        all_tailed_itineraries.append(tailed_itinerary)

        # Initialize the new columns with NaN
        tailed_itinerary['SOC_charging_start'] = np.nan
        tailed_itinerary['SOC_charging_end'] = np.nan

        # Build a mask for rows where charging occurs (start time is not NaN)
        mask_charging = ~tailed_itinerary['Charging_Start_Time'].isna()

        # 4) For rows with charging:
        #    - SOC_charging_end = SOC_Trip_end (final SOC after the charge)
        #    - SOC_charging_start = (SOC_Trip_start - trip fraction),
        #      i.e., battery fraction immediately after the trip but before charging.

        tailed_itinerary.loc[mask_charging, 'SOC_charging_end'] = \
            tailed_itinerary.loc[mask_charging, 'SOC']

        tailed_itinerary.loc[mask_charging, 'SOC_charging_start'] = \
            tailed_itinerary.loc[mask_charging, 'SOC_Trip_end']

        tailed_itinerary.loc[mask_charging, 'SOC_Trip_end'] = \
            tailed_itinerary.loc[mask_charging, 'SOC_Trip_end']

        # Concatenate all the individual DataFrames into one final DataFrame
    final_df = pd.concat(all_tailed_itineraries, ignore_index=True)
    final_df.name = f"final_df_{day_types}_{'_'.join(states)}"
    final_df["PERSONID"] = 1
    final_df['TripOrder'] = final_df.groupby('HOUSEID').cumcount() + 1
    final_df.drop(["date", "SOC", "Battery Capacity", "PERSONID"], axis=1, inplace=True)
    return final_df

def emvt_charging_plot(df):
    data_charging_summary, data_charging_grouped = process.plot_days_between_charges(df, min_energy=10, min_days=-1, max_days=15, excluded_makes=[])
    charging_demand_curve = process.create_charging_demand_curve(data_charging_summary)
    charging_demand_curve_agg = process.create_charging_demand_curve_agg(data_charging_summary)
    process.plot_charging_demand_curve(charging_demand_curve)
    process.plot_charging_demand_curve(charging_demand_curve_agg)


def charging_pipeline(final_df, days=None, delay_charging=False):
    start_time = time.time()

    if days is None:
        days = list(range(1, 4))  # Default to selecting days 1-3

    # Extract state name dynamically from the input DataFrame
    state_name = final_df["HHSTATE"].unique()[0] if "HHSTATE" in final_df.columns else "Unknown"

    # Step 1: Process initial data
    step_start = time.time()
    print("Step 1: Adding order column and assigning days of the week...")
    final_df = process.add_order_column(final_df)
    final_df_days = process.assign_days_of_week(final_df)
    print(f"Step 1 completed in {time.time() - step_start:.2f} seconds.")

    # Step 2: Identify charging sessions
    step_start = time.time()
    print("Step 2: Identifying charging sessions...")
    final_df_days = process.identify_charging_sessions(final_df_days)
    final_df_days, final_df_days_grouped = process.calculate_days_between_charges_synt(final_df_days)
    print(f"Step 2 completed in {time.time() - step_start:.2f} seconds.")

    # Step 3: Map destinations and determine charging level/speed
    step_start = time.time()
    print("Step 3: Mapping destinations and determining charging levels...")
    final_df_days = process.map_whytrp1s_to_destination(final_df_days)
    final_df_days["Charging_Level"] = final_df_days.apply(process.determine_charging_level, axis=1)
    final_df_days["Charging_Speed"] = final_df_days["Charging_Level"].apply(process.determine_charging_speed)
    print(f"Step 3 completed in {time.time() - step_start:.2f} seconds.")

    # Step 4: Battery processing and charging times
    step_start = time.time()
    print("Step 4: Calculating battery-related metrics and charging times...")
    final_df_days["batt_kwh"] = final_df_days["Battery Capacity"].apply(process.batt_kwh)
    final_df_days = process.calculate_charging_times(final_df_days)
    print(f"Step 4 completed in {time.time() - step_start:.2f} seconds.")

    # Step 5: Calculate trip energy and charging energy
    step_start = time.time()
    print("Step 5: Calculating trip and charging energy...")
    final_df_days = process.calculate_trip_energy(final_df_days)
    final_df_days = process.calculate_charging_energy(final_df_days)
    print(f"Step 5 completed in {time.time() - step_start:.2f} seconds.")

    # Step 6: Assign day numbers
    step_start = time.time()
    print("Step 6: Assigning day numbers...")
    final_df_days = process.assign_day_numbers(final_df_days)
    print(f"Step 6 completed in {time.time() - step_start:.2f} seconds.")

    # Step 7: Create weekly charging demand curve
    step_start = time.time()
    print("Step 7: Generating weekly charging demand curve...")
    weekly_demand_curve = process.create_weekly_charging_demand(final_df_days, delay_charging=delay_charging)
    print(f"Step 7 completed in {time.time() - step_start:.2f} seconds.")

    # Step 8: Filter demand curve by selected days and visualize
    step_start = time.time()
    print("Step 8: Filtering demand curve and visualizing results...")
    weekly_demand_curve = weekly_demand_curve[weekly_demand_curve["Day"].isin(days)]
    process.plot_charging_synt(weekly_demand_curve, state_name=state_name, delay_charging=delay_charging)
    print(f"Step 8 completed in {time.time() - step_start:.2f} seconds.")

    print(f"Pipeline completed successfully in {time.time() - start_time:.2f} seconds!")
    final_df_days["Charged_Energy"].fillna(0, inplace=True)
    return final_df_days, weekly_demand_curve


def generate_itineraries_for_selected_states(day_types="all", tile=3, state_counts=None, excluded_states=None, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json"):
    # Extract all unique states from state_counts
    all_states = state_counts['HHSTATE'].unique()

    # Remove excluded states if provided
    if excluded_states:
        states_to_run = [state for state in all_states if state not in excluded_states]
    else:
        states_to_run = all_states

    # Initialize a dictionary to store DataFrames for each state
    state_itineraries = {}

    # Iterate through each state to run
    for state in states_to_run:
        print(f"Processing state: {state}")
        try:
            # Generate itineraries for the current state
            df = generate_itineraries(
                day_types=day_types,
                tile=tile,
                states=[state],  # Pass the state
                state_counts=state_counts,
                electricity_price_file=electricity_price_file
            )
            # Save the result in the dictionary
            state_itineraries[state] = df
        except Exception as e:
            print(f"Error processing state {state}: {e}")

    print("Completed generating itineraries for selected states.")
    return state_itineraries


def run_itineraries_for_weeks(df_chagring=None, max_days=7, chunk=7, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=None):
    day_types = "eVMT"
    tile = 1
    states = ["CA"]
    step_start = time.time()
    # Generate a list of week intervals (7, 14, ..., up to max_days)
    week_intervals = list(range(chunk, max_days + 1, chunk))

    # Initialize dictionary to store results
    dict_of_dfs = {}

    for days in week_intervals:
        start_time = time.time()
        print(f"Running for {days} days...")

        # Unpack only the first returned value
        final_df_evmt = generate_itineraries_evmt_by_days(day_types=day_types, tile=tile, states=states, days_to_select=days, electricity_price_file=electricity_price_file, state_name=state_name_abrv)

        # Remove duplicate vehicle entries from data_charging
        data_charging_unique = df_chagring.sort_values("vehicle_model").drop_duplicates(subset=["vehicle_name"], keep="first")

        # Merge with vehicle model data
        final_df_evmt = pd.merge(final_df_evmt, data_charging_unique[["vehicle_name", "vehicle_model"]],
                                 left_on="VEHID", right_on="vehicle_name", how="left")

        # Store in dictionary
        dict_of_dfs[days] = final_df_evmt
        print(f"Run for {days} completed in {time.time() - start_time:.2f} seconds!")

    print(f" Process completed in {time.time() - step_start:.2f} seconds.")
    return dict_of_dfs


def run_itineraries_for_weeks_delay(df_chagring=None, max_days=7, chunk=7, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=None):
    day_types = "eVMT"
    tile = 1
    states = ["CA"]
    step_start = time.time()
    # Generate a list of week intervals (7, 14, ..., up to max_days)

    week_intervals = list(range(chunk, max_days + 1, chunk))

    # Initialize dictionary to store results
    dict_of_dfs = {}

    for days in week_intervals:
        start_time = time.time()
        print(f"Running for {days} days...")

        # Unpack only the first returned value
        final_df_evmt = generate_itineraries_evmt_by_days_delay(day_types=day_types, tile=tile, states=states, days_to_select=days, electricity_price_file=electricity_price_file, state_name=state_name_abrv)

        # Remove duplicate vehicle entries from data_charging
        data_charging_unique = df_chagring.sort_values("vehicle_model").drop_duplicates(subset=["vehicle_name"], keep="first")

        # Merge with vehicle model data
        final_df_evmt = pd.merge(final_df_evmt, data_charging_unique[["vehicle_name", "vehicle_model"]],
                                 left_on="VEHID", right_on="vehicle_name", how="left")

        # Store in dictionary
        dict_of_dfs[days] = final_df_evmt
        print(f"Run for {days} completed in {time.time() - start_time:.2f} seconds!")

    print(f" Process completed in {time.time() - step_start:.2f} seconds.")
    return dict_of_dfs


def selecting_chuncks(input_df):
    df_list = []
    # Process each dictionary key (each key contains a DataFrame)
    for key, df in input_df.items():
        # Ensure it's a DataFrame
        if isinstance(df, pd.DataFrame):
            # Check if required columns exist
            required_columns = {"Year", "Month", "Day", "HOUSEID", "VEHID"}
            if not required_columns.issubset(df.columns):
                print(f"Skipping key {key}: Missing columns {required_columns - set(df.columns)}")
                continue

            # Convert 'Year', 'Month', and 'Day' into a datetime column for filtering
            df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")

            # Drop rows where date could not be parsed
            df = df.dropna(subset=["date"])

            # Compute the first date per HOUSEID & VEHID
            min_dates = df.groupby(["HOUSEID", "VEHID"])["date"].transform("min")

            # Filter trips within the first 7 days of each vehicle's recorded trips
            df_filtered = df[df["date"] < (min_dates + pd.Timedelta(days=7))].copy()

            # Add new column to track which dictionary key this data came from
            df_filtered["source_key"] = key

            # Append filtered data to list
            df_list.append(df_filtered)

    # Ensure df_list is not empty before concatenating
    if df_list:
        charging_df = pd.concat(df_list, ignore_index=True)
    else:
        print("No valid data found for any vehicles in the first 7 days.")
    return charging_df


def aggregate_energy_demand(df, interval=15):
    df.loc[~df["Charging_Start_Time"].isna(), "charging_indicator"] = 1
    df.loc[df["Charging_Start_Time"].isna(), "charging_indicator"] = 0
    df = df[df["charging_indicator"] == 1]

    # Ensure Charging_Start_Time and Charging_End_Time are in HHMM format
    df["Charging_Start_Time"] = df["Charging_Start_Time"].astype(int).astype(str).str.zfill(4)
    df["Charging_End_Time"] = df["Charging_End_Time"].astype(int).astype(str).str.zfill(4)

    # Function to parse HHMM or HMM format into hours and minutes
    def parse_hhmm(time_str):
        time_str = str(time_str)
        if len(time_str) == 3:
            return int(time_str[0]), int(time_str[1:])
        elif len(time_str) == 4:
            return int(time_str[:2]), int(time_str[2:])
        return None, None

    # Apply function to extract hours and minutes
    df[["start_hour", "start_minute"]] = df["Charging_Start_Time"].apply(lambda x: pd.Series(parse_hhmm(x)))
    df[["end_hour", "end_minute"]] = df["Charging_End_Time"].apply(lambda x: pd.Series(parse_hhmm(x)))

    # Convert to total minutes
    df["start_time"] = df["start_hour"] * 60 + df["start_minute"]
    df["end_time"] = df["end_hour"] * 60 + df["end_minute"]

    # Drop invalid rows
    df = df.dropna(subset=["start_time", "end_time"])

    # Ensure day-month values are valid
    valid_days_per_month = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    df = df[df.apply(lambda row: row["Day"] <= valid_days_per_month[row["Month"]], axis=1)]

    # Convert Month and Day into a pseudo-date
    df["pseudo_date"] = pd.to_datetime(df[["Month", "Day"]].assign(Year=2023), errors="coerce")

    # Compute the first date per vehicle
    df["min_pseudo_date_per_vehicle"] = df.groupby(["HOUSEID", "VEHID"])["pseudo_date"].transform("min")
    df["relative_day"] = (df["pseudo_date"] - df["min_pseudo_date_per_vehicle"]).dt.days

    # Create a time index based on the selected interval
    time_intervals = np.arange(0, 24 * 60, interval)

    # Aggregate charging demand
    result_list = []
    for (scenario, relative_day), group in df.groupby(["source_key", "relative_day"]):
        demand_series = pd.Series(0.0, index=time_intervals, name="Energy_Demand_kWh")

        for _, row in group.iterrows():
            start_idx = np.searchsorted(time_intervals, row["start_time"])
            end_idx = np.searchsorted(time_intervals, row["end_time"], side="right")

            if row["end_time"] < row["start_time"]:  # Charging crosses midnight
                # Part 1: From start_time to midnight
                end_of_day_idx = np.searchsorted(time_intervals, 1440, side="right")
                duration_part1 = max(end_of_day_idx - start_idx, 1)
                demand_series.iloc[start_idx:end_of_day_idx] += row["Charging_kwh"] * (duration_part1 / (end_idx - start_idx + 1440))

                # Part 2: From midnight to end_time on the next day
                start_of_next_day_idx = np.searchsorted(time_intervals, 0, side="right")
                duration_part2 = max(end_idx - start_of_next_day_idx, 1)
                demand_series.iloc[start_of_next_day_idx:end_idx] += row["Charging_kwh"] * (duration_part2 / (end_idx - start_idx + 1440))
            else:
                # Normal case: Charging does not cross midnight
                if start_idx < len(time_intervals) and end_idx > 0 and end_idx > start_idx:
                    demand_series.iloc[start_idx:end_idx] += row["Charging_kwh"] / (end_idx - start_idx)

        # Convert index into a relative day representation
        demand_series.index = pd.to_timedelta(demand_series.index, unit="m") + pd.to_timedelta(relative_day, unit="D")
        demand_series = demand_series.reset_index()
        demand_series.columns = ["Time_Offset", "Energy_Demand_kWh"]
        demand_series["Scenario"] = scenario
        demand_series["Relative_Day"] = relative_day
        result_list.append(demand_series)

    return pd.concat(result_list, ignore_index=True)


def process_actual_charging(actual_df, simulated_df, interval=15):
    actual_df = actual_df.sort_values(["Household", "year", "month", "day"])
    actual_df["start_time"] = pd.to_datetime(actual_df["start_time"], format="%m/%d/%y %H:%M", errors="coerce")
    actual_df["end_time"] = pd.to_datetime(actual_df["end_time"], format="%m/%d/%y %H:%M", errors="coerce")

    # Extract time-related components
    actual_df["Month"] = actual_df["start_time"].dt.month
    actual_df["Day"] = actual_df["start_time"].dt.day
    actual_df["start_time_minutes"] = actual_df["start_time"].dt.hour * 60 + actual_df["start_time"].dt.minute
    actual_df["end_time_minutes"] = actual_df["end_time"].dt.hour * 60 + actual_df["end_time"].dt.minute

    # Match households with simulated data
    simulated_households = simulated_df.rename(columns={"HOUSEID": "Household"})[["Household", "vehicle_name"]].drop_duplicates()
    actual_df = actual_df.merge(simulated_households, on=["Household", "vehicle_name"], how="inner")

    # Compute relative days
    actual_df["pseudo_date"] = actual_df["start_time"].dt.date
    actual_df["min_pseudo_date_per_vehicle"] = actual_df.groupby(["Household", "vehicle_name"])["pseudo_date"].transform("min")
    actual_df["relative_day"] = (pd.to_datetime(actual_df["pseudo_date"]) - pd.to_datetime(actual_df["min_pseudo_date_per_vehicle"])).dt.days

    # Keep first 7 days
    actual_df = actual_df[actual_df["relative_day"] < 7]
    # actual_df = actual_df[actual_df["relative_day"] < 28]
    # Create a time index based on the selected interval
    time_intervals = np.arange(0, 24 * 60, interval)

    # Aggregate charging demand
    result_list = []
    for relative_day, group in actual_df.groupby("relative_day"):
        demand_series = pd.Series(0, index=time_intervals, name="Energy_Demand_kWh")

        for _, row in group.iterrows():
            start_idx = np.searchsorted(time_intervals, row["start_time_minutes"])
            end_idx = np.searchsorted(time_intervals, row["end_time_minutes"], side="right")

            # Handle cases where charging session crosses midnight
            if row["end_time_minutes"] < row["start_time_minutes"]:
                # First part: Charging before midnight (same day)
                end_of_day_idx = np.searchsorted(time_intervals, 1440, side="right")
                duration_part1 = max(end_of_day_idx - start_idx, 1)
                demand_series.iloc[start_idx:end_of_day_idx] += row["total_energy"] * (duration_part1 / (end_idx - start_idx + 1440))

                # Second part: Charging after midnight (next day)
                start_of_next_day_idx = np.searchsorted(time_intervals, 0, side="right")
                duration_part2 = max(end_idx - start_of_next_day_idx, 1)
                demand_series.iloc[start_of_next_day_idx:end_idx] += row["total_energy"] * (duration_part2 / (end_idx - start_idx + 1440))
            else:
                # Normal case: Charging does not cross midnight
                if start_idx < end_idx and end_idx <= len(time_intervals):
                    energy_per_slot = row["total_energy"] / max(end_idx - start_idx, 1)
                    demand_series.iloc[start_idx:end_idx] += energy_per_slot

        demand_series.index = pd.to_timedelta(demand_series.index, unit="m") + pd.to_timedelta(relative_day, unit="D")
        demand_series = demand_series.reset_index()
        demand_series.columns = ["Time_Offset", "Energy_Demand_kWh"]
        demand_series["Relative_Day"] = relative_day
        result_list.append(demand_series)

    return pd.concat(result_list, ignore_index=True)


def plot_charging_demand(df_simulated, df_actual, num_scenarios=None):

    # Convert Time_Offset to total minutes for proper plotting
    df_simulated["Total_Minutes"] = df_simulated["Time_Offset"].dt.total_seconds() / 60
    df_actual["Total_Minutes"] = df_actual["Time_Offset"].dt.total_seconds() / 60

    # Select scenarios to plot
    unique_scenarios = df_simulated["Scenario"].unique()
    if num_scenarios is not None:
        selected_scenarios = unique_scenarios[:num_scenarios]  # Limit scenarios
    else:
        selected_scenarios = unique_scenarios  # Use all scenarios

    # Create a figure and axis
    plt.figure(figsize=(14, 6))

    # Plot simulated scenarios
    for scenario in selected_scenarios:
        group = df_simulated[df_simulated["Scenario"] == scenario]
        aggregated_demand = group.groupby("Total_Minutes")["Energy_Demand_kWh"].sum()
        plt.plot(aggregated_demand.index, aggregated_demand.values, label=f"Scenario {scenario} days")

    # Plot actual charging demand (always included)
    aggregated_actual_demand = df_actual.groupby("Total_Minutes")["Energy_Demand_kWh"].sum()
    plt.plot(aggregated_actual_demand.index, aggregated_actual_demand.values, linestyle="--", color="black", linewidth=2, label="Actual Demand")

    # Formatting
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Energy Demand (kWh)")
    plt.title("Aggregated Charging Demand Over a Week")

    # Adjust x-axis labels to show each day with hours
    tick_positions = range(0, 7 * 24 * 60, 12 * 60)  # Every 12 hours
    tick_labels = [f"Day {i // 1440 + 1}, {i % 1440 // 60:02d}:{i % 60:02d}" for i in tick_positions]
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)

    # Move legend outside the plot
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend

    # Show plot
    plt.show()


def compute_pairwise_error_metrics(simulated_df):
    # simulated_df =energy_demand_uncordinated.copy()
    scenarios = simulated_df["Scenario"].unique()
    num_scenarios = len(scenarios)

    # Initialize matrices for error metrics
    mae_matrix = np.zeros((num_scenarios, num_scenarios))
    rmse_matrix = np.zeros((num_scenarios, num_scenarios))
    mbe_matrix = np.zeros((num_scenarios, num_scenarios))
    mape_matrix = np.zeros((num_scenarios, num_scenarios))
    smape_matrix = np.zeros((num_scenarios, num_scenarios))
    r2_matrix = np.zeros((num_scenarios, num_scenarios))

    # Aggregate energy demand across relative days for each scenario
    aggregated_data = {}
    for scenario in scenarios:
        scenario_df = simulated_df[simulated_df["Scenario"] == scenario]
        aggregated_data[scenario] = scenario_df.groupby("Time_Offset")["Energy_Demand_kWh"].sum()

    # Compute pairwise errors
    for i, j in itertools.combinations_with_replacement(range(num_scenarios), 2):
        scenario_i = scenarios[i]
        scenario_j = scenarios[j]

        # Get aligned data with union of indices
        combined_index = aggregated_data[scenario_i].index.union(aggregated_data[scenario_j].index)

        aligned_i = aggregated_data[scenario_i].reindex(combined_index, fill_value=0)
        aligned_j = aggregated_data[scenario_j].reindex(combined_index, fill_value=0)

        # Compute error metrics
        mae = np.mean(np.abs(aligned_i - aligned_j))
        rmse = np.sqrt(np.mean((aligned_i - aligned_j) ** 2))
        mbe = np.mean(aligned_j - aligned_i)
        mape = np.mean(np.abs((aligned_i - aligned_j) / (aligned_i + 1e-1))) * 100  # Avoid division by zero
        smape = np.mean(2 * np.abs(aligned_i - aligned_j) / (np.abs(aligned_i) + np.abs(aligned_j) + 1e-8)) * 100

        # Compute R-squared
        r2 = r2_score(aligned_i, aligned_j)

        # Store symmetric results in matrices
        mae_matrix[i, j] = mae_matrix[j, i] = mae
        rmse_matrix[i, j] = rmse_matrix[j, i] = rmse
        mbe_matrix[i, j] = mbe_matrix[j, i] = mbe
        mape_matrix[i, j] = mape_matrix[j, i] = mape
        smape_matrix[i, j] = smape_matrix[j, i] = smape
        r2_matrix[i, j] = r2_matrix[j, i] = r2

    # Convert matrices to DataFrames
    mae_df = pd.DataFrame(mae_matrix, index=scenarios, columns=scenarios)
    rmse_df = pd.DataFrame(rmse_matrix, index=scenarios, columns=scenarios)
    mbe_df = pd.DataFrame(mbe_matrix, index=scenarios, columns=scenarios)
    mape_df = pd.DataFrame(mape_matrix, index=scenarios, columns=scenarios)
    smape_df = pd.DataFrame(smape_matrix, index=scenarios, columns=scenarios)
    r2_df = pd.DataFrame(r2_matrix, index=scenarios, columns=scenarios)

    return {"MAE": mae_df, "RMSE": rmse_df, "MBE": mbe_df, "MAPE (%)": mape_df, "SMAPE (%)": smape_df, "R-Squared": r2_df}




