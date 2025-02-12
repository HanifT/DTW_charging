import os
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np


# %%
def _print(string, disp=True):
    if disp:
        print(string)


def load(codex_path, verbose=False):
    t0 = time.time()

    # Getting file parts for codex

    codex_directory, codex_file = os.path.split(codex_path)

    # Loading codex file

    with open(codex_path, 'r') as file:

        codex = json.load(file)

    # Loading in data

    data = {}

    for item in codex:

        extension = item['file'].split('.')[-1]

        load_path = os.path.join(codex_directory, item['file'])

        if extension == 'csv':

            data[item['name']] = pd.read_csv(load_path, **item['kwargs'])

        elif (extension == 'xlsx') or (extension == 'xls'):

            data[item['name']] = pd.read_excel(load_path, **item['kwargs'])

        else:

            raise RawDataFileTypeException

    _print(f'Data loaded: {time.time() - t0:.4} seconds', disp=verbose)

    return data


class RawDataFileTypeException(Exception):
    pass


def plot_charging_demand(df, plot_type='total'):
    """
    Plots the charging demand for households with different charger types (Level 1 Only, Level 2 Only, Both Levels)
    at home and non-home locations.

    Parameters:
    - df (dict): Dictionary containing data frames for Level 1 Only, Level 2 Only, and Both Levels charging demand.
      Each data frame should be a dictionary with 'home' and 'non_home' keys.
    - plot_type (str): Type of plot. Options are 'total' for total energy demand or 'average' for average demand per vehicle.
    """
    # Unpack data frames
    level1_home, level1_non_home = df['Level 1 Only']
    level2_home, level2_non_home = df['Level 2 Only']
    both_home, both_non_home = df['Both Levels']

    # Calculate average if requested
    if plot_type == 'average':
        level1_home['total_energy'] /= level1_home['vehicle_name'].nunique()
        level1_non_home['total_energy'] /= level1_non_home['vehicle_name'].nunique()
        level2_home['total_energy'] /= level2_home['vehicle_name'].nunique()
        level2_non_home['total_energy'] /= level2_non_home['vehicle_name'].nunique()
        both_home['total_energy'] /= both_home['vehicle_name'].nunique()
        both_non_home['total_energy'] /= both_non_home['vehicle_name'].nunique()

    # Update the 'Label' values to show only three main categories
    level1_home['Label'] = 'Level 1 Only'
    level1_non_home['Label'] = 'Level 1 Only'
    level2_home['Label'] = 'Level 2 Only'
    level2_non_home['Label'] = 'Level 2 Only'
    both_home['Label'] = 'Both Levels'
    both_non_home['Label'] = 'Both Levels'

    # Concatenate all DataFrames with updated labels
    combined_df = pd.concat([
        level1_home.assign(Location='Home'),
        level1_non_home.assign(Location='Non-Home'),
        level2_home.assign(Location='Home'),
        level2_non_home.assign(Location='Non-Home'),
        both_home.assign(Location='Home'),
        both_non_home.assign(Location='Non-Home')
    ])

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Label', y='total_energy', hue='Location', data=combined_df)

    # Set plot labels
    plt.xlabel('Home Charger Type')
    plt.ylabel('Total Energy Demand (kWh) per Vehicle' if plot_type == 'total' else 'Average Energy Demand per Vehicle (kWh)')
    plt.title('Home vs Non-Home Charging Demand by Household Charger Type' if plot_type == 'total' else
              'Home vs Non-Home Average Charging Demand per Vehicle by Household Type')
    plt.xticks(rotation=45)
    plt.legend(title='Location')
    plt.tight_layout()
    plt.show()


def plot_days_between_charges(data, min_energy=2, min_days=0, max_days=10, excluded_makes=["Nissan", "Toyota"]):
    # Exclude specific vehicle makes
    data_filtered = data[~data["Make"].isin(excluded_makes)]

    # Convert 'start_time_ (local)' column to datetime format
    data_filtered['start_time_ (local)'] = pd.to_datetime(data_filtered['start_time_ (local)'], format='%m/%d/%y %H:%M')

    # Create a new column for the day of the year
    data_filtered['day_of_year'] = data_filtered['start_time_ (local)'].dt.dayofyear

    # Group by vehicle_name and day_of_year, then get the first charging session time for each day
    df_grouped = data_filtered.groupby(['vehicle_name', 'day_of_year']).first().reset_index()

    # Filter out sessions with less than the specified minimum energy
    df_grouped = df_grouped[df_grouped["total_energy"] > min_energy]

    # Sort by vehicle and time to ensure chronological order
    df_grouped = df_grouped.sort_values(by=['vehicle_name', 'start_time_ (local)'])

    # Calculate the number of days between each charging session for each vehicle
    df_grouped['days_between_charges'] = df_grouped.groupby('vehicle_name')['day_of_year'].diff()

    # Fill NaN values in 'days_between_charges' with 0, indicating the first session
    df_grouped['days_between_charges'] = df_grouped['days_between_charges'].fillna(0)

    # Filter based on days between charges
    df_grouped = df_grouped[(df_grouped['days_between_charges'] > min_days) & (df_grouped['days_between_charges'] < max_days)]

    # Plot box plot of days between charges grouped by vehicle model
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_grouped, x='Model', y='days_between_charges')

    # Customize plot labels and title
    plt.xticks(rotation=45)
    plt.xlabel("Vehicle Model")
    plt.ylabel("Days Between Charges")
    plt.title("Distribution of Days Between Charging Sessions by Vehicle Model")

    # Use tight layout to fit all elements within the figure
    plt.tight_layout()

    # Display the plot
    plt.show()

    return data_filtered, df_grouped


def add_order_column(df):
    # Use groupby and cumcount to create the 'order' column
    df['order'] = df.groupby(['HOUSEID', 'PERSONID', 'VEHID']).cumcount() + 1

    return df


def assign_days_of_week(df):
    # Define the days of the week
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Sort by HOUSEID, PERSONID, VEHID, and STRTTIME to ensure chronological order
    df = df.sort_values(by=['HOUSEID', 'PERSONID', 'VEHID', "order", "TripOrder"]).reset_index(drop=True)

    # Initialize an empty list to store the day of the week for each trip
    day_of_week_column = []

    # Initialize variables to track the previous STRTTIME and current day index for each unique vehicle
    prev_strttime = None
    day_index = 0

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Check if this row is the first entry for a new HOUSEID-PERSONID-VEHID combination
        if i == 0 or (
                row['HOUSEID'] != df.at[i - 1, 'HOUSEID'] or
                row['PERSONID'] != df.at[i - 1, 'PERSONID'] or
                row['VEHID'] != df.at[i - 1, 'VEHID']
        ):
            # Start a new week for this unique vehicle
            day_index = 0
            day_of_week_column.append(days_of_week[day_index])
        else:
            # Check if STRTTIME is less than the previous time, indicating a new day
            if row['STRTTIME'] <= prev_strttime:
                day_index = (day_index + 1) % 7  # Move to the next day, wrap around after Sunday

            # Append the current day of the week to the list
            day_of_week_column.append(days_of_week[day_index])

        # Update the previous STRTTIME to the current one
        prev_strttime = row['STRTTIME']

    # Add the day_of_week_column as a new column in the DataFrame
    df['days_of_week'] = day_of_week_column

    return df


def identify_charging_sessions(df):
    # Sort by HOUSEID, PERSONID, VEHID, and order to ensure chronological order within each group
    df = df.sort_values(by=['HOUSEID', 'PERSONID', 'VEHID', 'order']).reset_index(drop=True)

    # Calculate the difference in SOC within each unique vehicle group
    df['charging'] = df.groupby(['HOUSEID', 'PERSONID', 'VEHID'])['SOC'].diff().fillna(0) > 0

    return df


def calculate_days_between_charges_synt(df):
    # Define a mapping from days of the week to numeric values
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                   "Friday": 4, "Saturday": 5, "Sunday": 6}

    # Map 'days_of_week' to numeric values for easy calculation
    df['day_numeric'] = df['days_of_week'].map(day_mapping)

    # Sort by HOUSEID, PERSONID, VEHID, and 'order' to ensure chronological order within each group
    df = df.sort_values(by=['HOUSEID', 'PERSONID', 'VEHID', 'order']).reset_index(drop=True)

    # Initialize a list to store the days between charges for each row
    days_between_charges = []

    # Track the last charging day for each unique vehicle
    last_charging_day = {}

    # Iterate through each row
    for i, row in df.iterrows():
        # Create a unique key for each vehicle
        vehicle_key = (row['HOUSEID'], row['PERSONID'], row['VEHID'])

        # Check if this row is the first entry for the vehicle
        if vehicle_key not in last_charging_day:
            # Set days_between_charges to 0 for the first entry
            days_diff = 0
            last_charging_day[vehicle_key] = row['day_numeric']
        else:
            # For subsequent entries, check if this is a charging event
            if row['charging'] == True:
                # Calculate the difference in days, adjusting for week wrap-around
                days_diff = (row['day_numeric'] - last_charging_day[vehicle_key]) % 7
                # Update the last charging day for this vehicle
                last_charging_day[vehicle_key] = row['day_numeric']
            else:
                # If it's not a charging event, set days_diff to None
                days_diff = None

        # Append the calculated days_diff to the list
        days_between_charges.append(days_diff)

    # Add the 'days_between_charges' column to the DataFrame and drop the 'day_numeric' helper column
    df['days_between_charges'] = days_between_charges
    df = df.drop(columns=['day_numeric'])

    # Filter out None values and the first row (order == 1) for each unique vehicle for plotting
    df_filtered = df[(df['days_between_charges'].notna()) & (df['order'] != 1)]

    # Plot the box plot for days_between_charges grouped by battery capacity (BATTCAP)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x='BATTCAP', y='days_between_charges')
    plt.xlabel("Battery Capacity (kWh)")
    plt.ylabel("Days Between Charges")
    plt.title("Distribution of Days Between Charges by Battery Capacity")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

    return df, df_filtered


def create_charging_demand_curve_agg(df):
    # Step 1: Divide `day_of_year` into 14 periods
    df["Period_Day"] = (df["day_of_year"] - 1) // (365 // 14)  # Map day_of_year to one of 14 periods

    # Step 2: Round start_time and end_time to the nearest 30 minutes
    def round_to_half_hour(timestamp):
        dt = datetime.strptime(timestamp, "%m/%d/%y %H:%M")  # Parse the timestamp
        if dt.minute < 15:
            dt = dt.replace(minute=0)
        elif dt.minute < 45:
            dt = dt.replace(minute=30)
        else:
            dt = dt.replace(minute=0) + timedelta(hours=1)
        return dt

    df["Start_Rounded"] = df["start_time"].apply(round_to_half_hour)
    df["End_Rounded"] = df["end_time"].apply(round_to_half_hour)

    # Step 3: Assign charging speeds based on `energy[charge__type][type]`
    def assign_charging_speed(charge_type):
        if charge_type == "LEVEL_2":
            return 6.6
            # kW
        elif charge_type == "DC_FAST":
            return 150  # kW
        else:
            return 0  # No charging

    df["Charging_Speed"] = df["energy[charge_type][type]"].apply(assign_charging_speed)

    # Step 4: Initialize the demand curve for 14 days (48 periods/day)
    max_days = 14
    demand_curve = np.zeros(max_days * 48)

    # Step 5: Simulate charging sessions
    for _, row in df.iterrows():
        charging_speed = row["Charging_Speed"]
        total_energy = row["total_energy"]  # Energy required for charging (kWh)
        soc_remaining = total_energy  # Remaining energy to charge
        start_time = row["Start_Rounded"]
        end_time = row["End_Rounded"]
        current_day = row["Period_Day"]

        # Calculate start and end periods
        start_period = (start_time.hour * 2) + (1 if start_time.minute >= 30 else 0)
        end_period = (end_time.hour * 2) + (1 if end_time.minute >= 30 else 0)

        while soc_remaining > 0:
            if current_day >= max_days:  # Ignore days beyond the 14th period
                break

            # Get the start and end of the current day in terms of periods
            daily_start = current_day * 48
            daily_end = (current_day + 1) * 48

            # Simulate charging across periods
            for period in range(daily_start + start_period, daily_end):
                if soc_remaining <= 0:
                    break
                charge_time = 0.5  # 30 minutes
                charge_amount = min(soc_remaining, charging_speed * charge_time)
                demand_curve[period] += charge_amount
                soc_remaining -= charge_amount

            current_day += 1  # Move to the next day
            start_period = 0  # Reset to the start of the day

    # Step 6: Create a DataFrame for the demand curve
    demand_curve_df = pd.DataFrame({
        "Period": np.tile(range(48), max_days),
        "Day": np.repeat(range(max_days), 48),
        "Demand (kWh)": demand_curve
    })

    return demand_curve_df


def create_charging_demand_curve(df):
    # Step 1: Divide `day_of_year` into 14 periods
    df["Period_Day"] = (df["day_of_year"] - 1)  # Map day_of_year to one of 14 periods

    # Step 2: Round start_time and end_time to the nearest 30 minutes
    def round_to_half_hour(timestamp):
        dt = datetime.strptime(timestamp, "%m/%d/%y %H:%M")  # Parse the timestamp
        if dt.minute < 15:
            dt = dt.replace(minute=0)
        elif dt.minute < 45:
            dt = dt.replace(minute=30)
        else:
            dt = dt.replace(minute=0) + timedelta(hours=1)
        return dt

    df["Start_Rounded"] = df["start_time"].apply(round_to_half_hour)
    df["End_Rounded"] = df["end_time"].apply(round_to_half_hour)

    # Step 3: Assign charging speeds based on `energy[charge__type][type]`
    def assign_charging_speed(charge_type):
        if charge_type == "LEVEL_2":
            return 6.6  # kW
        elif charge_type == "DC_FAST":
            return 150  # kW
        else:
            return 0  # No charging

    df["Charging_Speed"] = df["energy[charge_type][type]"].apply(assign_charging_speed)

    # Step 4: Initialize the demand curve for 14 days (48 periods/day)
    max_days = 14
    demand_curve = np.zeros(max_days * 48)

    # Step 5: Simulate charging sessions
    for _, row in df.iterrows():
        charging_speed = row["Charging_Speed"]
        total_energy = row["total_energy"]  # Energy required for charging (kWh)
        soc_remaining = total_energy  # Remaining energy to charge
        start_time = row["Start_Rounded"]
        end_time = row["End_Rounded"]
        current_day = row["Period_Day"]

        # Calculate start and end periods
        start_period = (start_time.hour * 2) + (1 if start_time.minute >= 30 else 0)
        end_period = (end_time.hour * 2) + (1 if end_time.minute >= 30 else 0)

        while soc_remaining > 0:
            if current_day >= max_days:  # Ignore days beyond the 14th period
                break

            # Get the start and end of the current day in terms of periods
            daily_start = current_day * 48
            daily_end = (current_day + 1) * 48

            # Simulate charging across periods
            for period in range(daily_start + start_period, daily_end):
                if soc_remaining <= 0:
                    break
                charge_time = 0.5  # 30 minutes
                charge_amount = min(soc_remaining, charging_speed * charge_time)
                demand_curve[period] += charge_amount
                soc_remaining -= charge_amount

            current_day += 1  # Move to the next day
            start_period = 0  # Reset to the start of the day

    # Step 6: Create a DataFrame for the demand curve
    demand_curve_df = pd.DataFrame({
        "Period": np.tile(range(48), max_days),
        "Day": np.repeat(range(max_days), 48),
        "Demand (kWh)": demand_curve
    })

    return demand_curve_df


def plot_charging_demand_curve(demand_curve_df):
    # Filter the data to include only 14 days
    demand_curve_df = demand_curve_df[demand_curve_df["Day"] < 13]

    # Create custom x-tick labels for 24-hour format across 7 days
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Generate x-tick positions and labels for the entire week
    x_ticks = np.arange(0, 48 * 14, 48)  # 48 periods per day
    x_tick_labels = [f"{day}" for day in days]
    # Create custom x-tick labels for 24-hour format across 14 days
    # days = [f"Day {i + 1}" for i in range(14)]
    # x_ticks = np.arange(0, 48 * 14, 48)  # 48 periods per day
    # x_tick_labels = days

    # Plot the demand curve
    plt.figure(figsize=(14, 7))
    plt.plot(
        demand_curve_df["Period"] + demand_curve_df["Day"] * 48,
        demand_curve_df["Demand (kWh)"],
        label="Charging Demand",
        color="blue"
    )

    # Set the x-axis ticks and labels
    plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=45)

    # Add gridlines to separate days
    for tick in x_ticks:
        plt.axvline(x=tick, color='grey', linestyle='--', linewidth=0.5)

    # Add labels and title
    plt.xlabel("Day of the Week")
    plt.ylabel("Charging Demand (kWh)")
    plt.title("Weekly Charging Demand Curve")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()


def map_whytrp1s_to_destination(df, column="WHYTRP1S"):
    whytrp1s_mapping = {
        1: "Home",
        10: "Work",
        20: "School",
        30: "Medical",
        40: "Shopping",
        50: "Social",
        70: "Transport",
        80: "Meals",
        97: "Other"
    }
    # Map the WHYTRP1S values to a new column 'Destination'
    df["Destination"] = df[column].map(whytrp1s_mapping)
    return df


def determine_charging_level(row):
    if row["charging"] and row["Destination"] in ["Home", "Work", "School"]:
        return "Level_2"
    elif row["charging"]:
        return "DC_Fast"
    else:
        return None  # No charging session

    # Define a function to assign charging speed


def determine_charging_speed(charging_level):
    if charging_level == "Level_2":
        return 6.6  # Level 2 charging speed
    elif charging_level == "DC_Fast":
        return 150  # DC Fast charging speed
    else:
        return None  # No charging speed if there's no charging level


def batt_kwh(bat_cap):
    return bat_cap * 2.77778e-7


def calculate_charging_times(df):
    # Ensure data is sorted by HOUSEID, PERSONID, VEHID, order, and STRTTIME for processing
    df = df.sort_values(by=["HOUSEID", "PERSONID", "VEHID", "order", "STRTTIME"]).reset_index(drop=True)

    # Initialize empty columns
    df["Charging_Start_Time"] = None
    df["Charging_End_Time"] = None

    # Iterate through each unique combination of HOUSEID, PERSONID, and VEHID
    for key, group in df.groupby(["HOUSEID", "PERSONID", "VEHID"]):
        group_indices = group.index  # Indices of the group for assigning values
        first_start_time = int(group.iloc[0]["STRTTIME"])  # Store the first row's STRTTIME

        for i, idx in enumerate(group_indices):
            # Skip rows where the Charging Level is "None"
            if group.loc[idx, "Charging_Level"] == None:
                continue

            # Assign Charging Start Time as ENDTIME
            df.loc[idx, "Charging_Start_Time"] = group.loc[idx, "ENDTIME"]

            if i < len(group_indices) - 1:
                # For non-last rows, Charging End Time is the next STRTTIME
                df.loc[idx, "Charging_End_Time"] = group.loc[group_indices[i + 1], "STRTTIME"]
            else:
                # For the last row in the group, Charging End Time is the next morning's first STRTTIME
                df.loc[idx, "Charging_End_Time"] = str(
                    (2400 + first_start_time) % 2400
                ).zfill(4)  # Format to 4 digits (e.g., 800 -> "0800")

            # Check for DC_Fast chargers and limit the duration to 1 hour
            if group.loc[idx, "Charging_Level"] == "DC_Fast":
                start_time = int(df.loc[idx, "Charging_Start_Time"])
                end_time = int(df.loc[idx, "Charging_End_Time"])

                if (end_time - start_time) > 100 or (start_time > end_time):  # Check if duration > 1 hour
                    df.loc[idx, "Charging_End_Time"] = str((start_time + 100) % 2400).zfill(4)

            # Handle cases where Charging End Time is smaller than Charging Start Time (next day)
            start_time = int(df.loc[idx, "Charging_Start_Time"])
            end_time = int(df.loc[idx, "Charging_End_Time"])

            if end_time < start_time:
                df.loc[idx, "Charging_End_Time"] = str((start_time + (2400 - start_time) + end_time) % 2400).zfill(4)

    return df


def calculate_charging_energy(df):
    # Ensure the DataFrame is sorted to allow correct computation
    df = df.sort_values(by=["HOUSEID", "PERSONID", "VEHID", "order", "TripOrder"]).reset_index(drop=True)

    # Initialize the new column
    df["Charged_Energy"] = np.nan

    # Iterate through each group of HOUSEID, PERSONID, and VEHID
    for key, group in df.groupby(["HOUSEID", "PERSONID", "VEHID"]):
        # Iterate over rows in the group
        for i in range(1, len(group)):
            current_idx = group.index[i]
            previous_idx = group.index[i - 1]

            # Perform the calculation only if the current row is a charging event
            if group.loc[current_idx, "charging"]:
                # Previous SOC * Battery Capacity
                previous_soc_energy = group.loc[previous_idx, "SOC"] * group.loc[previous_idx, "Battery Capacity"]

                current_soc_energy = group.loc[current_idx, "SOC"] * group.loc[current_idx, "Battery Capacity"]

                current_trip_energy = group.loc[current_idx, "Trip_Energy"]

                # Energy charged = Previous SOC Energy + Trip Energy
                charged_energy = current_soc_energy - (previous_soc_energy - current_trip_energy)

                # Assign the value to the current row
                df.loc[current_idx, "Charged_Energy"] = max(0, charged_energy) / 3.6e6  # Ensure non-negative values

    return df


def assign_day_numbers(df):
    # Ensure the DataFrame is sorted to detect day changes
    df = df.sort_values(by=["HOUSEID", "PERSONID", "VEHID", "order", "TripOrder"]).reset_index(drop=True)

    # Initialize the new column
    df["Day_Number"] = np.nan

    # Iterate through each group of HOUSEID, PERSONID, and VEHID
    for key, group in df.groupby(["HOUSEID", "PERSONID", "VEHID"]):
        # Track the current day number
        current_day_number = 1
        previous_day = None

        for idx in group.index:
            # Assign the current day number if the day_of_week changes, increment day number
            current_day = group.loc[idx, "days_of_week"]
            if previous_day is not None and current_day != previous_day:
                current_day_number += 1

            # Assign the day number to the row
            df.loc[idx, "Day_Number"] = current_day_number
            previous_day = current_day

    return df


def calculate_trip_energy(df, vmt_column="TRPMILES", consumption_factor=782.928):
    # Convert VMT_Mile to meters (1 mile = 1609.34 meters)
    df["Trip_Distance_Meters"] = df[vmt_column] * 1609.34

    # Calculate trip energy in Joules
    df["Trip_Energy"] = df["Trip_Distance_Meters"] * consumption_factor  # in jule

    return df


def create_weekly_charging_demand(df, delay_charging=False):
    df = df[df["Day_Number"] <= 15]
    # Filter data for valid charging sessions
    charging_sessions = df[pd.notna(df["Charging_Level"])].copy()

    # Define the rounding function
    def round_to_half_hour(time):
        time = int(time)  # Ensure the time is an integer
        hour = time // 100
        minute = time % 100

        if minute < 15:
            minute = 0
        elif minute < 45:
            minute = 30
        else:
            minute = 0
            hour += 1

        if hour == 24:  # Handle overflow to the next day
            hour = 0

        return hour * 100 + minute  # Return in HHMM format

    # Map day names to integers
    day_mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    # Convert 'days_of_week' to integers
    charging_sessions["days_of_week"] = charging_sessions["days_of_week"].map(day_mapping)

    # Ensure columns are integers before applying the rounding function
    charging_sessions["Charging_Start_Time"] = charging_sessions["Charging_Start_Time"].astype(int)

    # Apply the rounding function
    charging_sessions["Start_Rounded"] = charging_sessions["Charging_Start_Time"].apply(round_to_half_hour)

    if delay_charging:
        # Delay charging sessions starting between 8 PM and 6 AM to midnight
        def delay_to_midnight(start_time):
            hour = start_time // 100
            if hour >= 20 or hour < 6:  # Check if it's between 8 PM and 6 AM
                return 0  # Delay to midnight (00:00 in HHMM format)
            return start_time

        charging_sessions["Start_Rounded"] = charging_sessions["Start_Rounded"].apply(delay_to_midnight)

    # Initialize the demand curve for 14 days (48 periods/day)
    max_days = 15
    weekly_demand = np.zeros(max_days * 48)

    # Iterate through each charging session
    for _, row in charging_sessions.iterrows():
        charging_speed = row["Charging_Speed"]  # kWh per hour
        charged_energy = row["Charged_Energy"]  # Energy to charge (kWh)
        start_time = int(row["Start_Rounded"])
        soc_remaining = charged_energy  # Remaining energy to be charged
        current_day = int(row["Day_Number"]) - 1  # Adjust Day_Number to 0-based index
        start_period = (start_time // 100) * 2 + (1 if start_time % 100 >= 30 else 0)

        # Simulate charging across periods
        while soc_remaining > 0:
            # Ensure we don't go beyond the allocated days
            if current_day >= max_days:
                break

            # Calculate periods for the current day
            daily_start = current_day * 48
            daily_end = (current_day + 1) * 48

            for period in range(daily_start + start_period, daily_end):
                if soc_remaining <= 0:
                    break
                charge_time = 0.5  # Each period is 30 minutes
                charge_amount = min(soc_remaining, charging_speed * charge_time)  # kWh charged in this period
                weekly_demand[period] += charge_amount
                soc_remaining -= charge_amount

            current_day += 1  # Move to the next day
            start_period = 0  # Start from the beginning of the day

    # Create a DataFrame for the demand curve
    demand_curve = pd.DataFrame({
        "Period": np.tile(range(48), max_days),
        "Day": np.repeat(range(1, max_days + 1), 48),
        "Demand (kWh)": weekly_demand
    })

    return demand_curve


def plot_charging_synt(weekly_demand_curve, state_name, delay_charging):
    """
    Plots the weekly charging demand curve.

    Parameters:
    - weekly_demand_curve: DataFrame containing the demand curve data.
    - state_name: Name of the state.
    - delay_charging: Boolean indicating if delayed charging was applied.
    """
    # Filter the data to include only relevant days
    weekly_demand_curve_filtered = weekly_demand_curve[weekly_demand_curve["Day"] < 15]
    weekly_demand_curve_filtered["Day"] = weekly_demand_curve_filtered["Day"] - 7

    # Create custom x-tick labels for 24-hour format across days
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]

    # Generate x-tick positions and labels for the entire week
    x_ticks = np.arange(48, 48 * 9, 48)  # 48 periods per day
    x_tick_labels = [f"{day}" for day in days]

    plt.figure(figsize=(12, 6))

    # Plot the demand curve
    plt.plot(
        weekly_demand_curve_filtered["Period"] + weekly_demand_curve_filtered["Day"] * 48,
        weekly_demand_curve_filtered["Demand (kWh)"],
        label="Charging Demand",
        color="blue"
    )

    # Set the x-axis ticks and labels
    plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=45)

    # Add gridlines to separate days
    for tick in x_ticks:
        plt.axvline(x=tick, color='grey', linestyle='--', linewidth=0.5)

    # Add labels and title with state name and delay charging status
    title = f"Weekly Charging Demand Curve for {state_name}"
    title += " (Delayed Charging Enabled)" if delay_charging else " (Delayed Charging Disabled)"

    plt.xlabel("Day of the Week")
    plt.ylabel("Charging Demand (kWh)")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_battery_market_share_pie(battery_capacities, probabilities):
    """
    Plots a pie chart for the market share of battery capacities and displays the average of the rest.

    Parameters:
    - battery_capacities: List of battery capacities in joules.
    - probabilities: List of probabilities corresponding to the battery capacities.
    """
    # Convert battery capacities to kWh
    battery_capacities_kWh = [capacity / 3.6e6 for capacity in battery_capacities]  # Convert to kWh

    # Calculate the average of the remaining battery capacities (excluding the last element)
    avg_remaining = sum(battery_capacities_kWh[:-1]) / len(battery_capacities_kWh[:-1])

    # Create labels for the pie chart
    labels = [f"{int(cap)} kWh" for cap in battery_capacities_kWh]

    # Update the label of the last element with the average of the remaining
    labels[-1] += f"\n(Average of Rest)"

    # Plot the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

    # Add a title
    plt.title("Battery Capacity Market Share", fontsize=14)

    # Display the plot
    plt.tight_layout()
    plt.show()
