import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import process
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
from retry_requests import retry
# %%
# Preprocessing the temp, you can use the saved data


def standardize_datetime_format(datetime_column):
    return datetime_column.str.replace(r"\.\d+", "", regex=True)


def get_temperature_for_time(latitude, longitude, target_time):
    """
    Retrieve temperature data for a given latitude, longitude, and specific time using Open-Meteo Archive API,
    adjusted for Los Angeles time.
    """
    # Setup cache and retry logic
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Handle target_time: Localize or Convert to Los Angeles time
    target_time_dt = pd.to_datetime(target_time)
    if target_time_dt.tzinfo is None:
        # If naive, localize to Los Angeles timezone
        target_time_dt = target_time_dt.tz_localize("America/Los_Angeles")
    else:
        # If already timezone-aware, convert to Los Angeles timezone
        target_time_dt = target_time_dt.tz_convert("America/Los_Angeles")

    # Extract date range from Los Angeles time
    start_date = target_time_dt.strftime("%Y-%m-%d")
    end_date = target_time_dt.strftime("%Y-%m-%d")

    # API parameters
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "America/Los_Angeles"
    }

    # Fetch response
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    # Create DataFrame for hourly data
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("America/Los_Angeles"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("America/Los_Angeles"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly_temperature_2m
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Normalize timestamps
    hourly_dataframe["date"] = pd.to_datetime(hourly_dataframe["date"]).dt.floor("T")
    target_time_dt = target_time_dt.floor("T")

    # Check if the target_time is in the DataFrame
    if (hourly_dataframe["date"] == target_time_dt).any():
        temperature_row = hourly_dataframe.loc[hourly_dataframe["date"] == target_time_dt]
        temperature = temperature_row["temperature_2m"].iloc[0]
        return temperature
    else:
        print(f"Time {target_time_dt} not found in hourly data.")
        return None


def add_temperature_to_trips(data_trips_temp):
    start_time = time.time()  # Record start time
    # Step 2: Ensure the column is treated as a string
    data_trips_temp["start_time (local)"] = data_trips_temp["start_time (local)"].astype(str)

    # Step 3: Replace '-07:00' with '-08:00' in the string representation
    data_trips_temp["start_time (local)"] = data_trips_temp["start_time (local)"].str.replace("-07:00", "-08:00")

    # Step 2: Standardize datetime format
    def standardize_datetime_format(datetime_column):
        return datetime_column.str.replace(r"\.\d+", "", regex=True)

    data_trips_temp["start_time (local)"] = standardize_datetime_format(data_trips_temp["start_time (local)"])
    data_trips_temp["start_time (local)"] = data_trips_temp["start_time (local)"].str.strip()

    # Step 3: Convert to datetime
    data_trips_temp["start_time (local)"] = pd.to_datetime(
        data_trips_temp["start_time (local)"], errors="coerce"
    )
    data_trips_temp["rounded_time"] = data_trips_temp["start_time (local)"].dt.round("H")
    data_trips_temp = data_trips_temp.reset_index(drop=True)

    # Convert timestamps to America/Los_Angeles timezone with DST handling

    # Add a new column for temperature
    data_trips_temp["temperature"] = None  # Initialize with None

    # # Iterate through each row to fetch temperature

    # Iterate through each row
    for index, row in data_trips_temp.iterrows():
        # Skip if the temperature is already populated
        if pd.notna(row["temperature"]):
            continue
        try:
            # Extract values for the function
            latitude = row["location[start][latitude]"]
            longitude = row["location[start][longitude]"]
            target_time = row["rounded_time"]  # Use rounded time
            # Call the function and fetch temperature
            temperature = get_temperature_for_time(latitude, longitude, target_time)
            # Check if temperature is None and set a placeholder value (1000) for missing data
            if temperature is None:
                temperature = 1000  # Set large placeholder value for missing data

            # Assign the result to the new column
            data_trips_temp.at[index, "temperature"] = temperature
        except Exception as e:
            # Log the error and assign placeholder value (1000)
            print(f"Error processing row {index}: {e}")
            data_trips_temp.at[index, "temperature"] = 1000
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.2f} seconds")
    return data_trips_temp


def process_temperature_in_chunks(data_trips_temp, chunk_size=1000):
    """
    Process rows with missing temperatures in chunks.
    Each run processes up to `chunk_size` rows with missing temperature values.
    """
    # Filter rows where temperature is missing
    rows_to_process = data_trips_temp[data_trips_temp["temperature"].isna()]

    # If no rows are missing temperature, return the DataFrame
    if rows_to_process.empty:
        print("All rows are processed. No rows with missing temperature remain.")
        return data_trips_temp

    # Limit to the first `chunk_size` rows
    rows_to_process = rows_to_process.iloc[:chunk_size]

    print(f"Processing {len(rows_to_process)} rows with missing temperature...")

    # Process the selected rows and update the DataFrame
    updated_rows = add_temperature_to_trips(rows_to_process)

    # Merge updated rows back into the main DataFrame based on the 'id' column
    data_trips_temp = data_trips_temp.set_index("id")
    updated_rows = updated_rows.set_index("id")
    data_trips_temp.update(updated_rows)  # This updates the rows in the main DataFrame
    data_trips_temp = data_trips_temp.reset_index()

    print(f"Processed {len(rows_to_process)} rows.")
    return data_trips_temp


def prepare_temp(df):
    # Step 1: Prepare the data
    data_trips_temp = df[["id", "start_time (local)", "location[start][longitude]", "location[start][latitude]"]]
    data_trips_temp["temperature"] = None
    data_trips_temp["start_time (local)"] = data_trips_temp["start_time (local)"].astype(str)
    data_trips_temp["start_time (local)"] = data_trips_temp["start_time (local)"].str.replace("-07:00", "-08:00")
    # Standardize datetime format
    data_trips_temp["start_time (local)"] = standardize_datetime_format(data_trips_temp["start_time (local)"])
    data_trips_temp["start_time (local)"] = data_trips_temp["start_time (local)"].str.strip()
    # Convert to datetime and round
    data_trips_temp["start_time (local)"] = pd.to_datetime(data_trips_temp["start_time (local)"], errors="coerce")
    data_trips_temp["rounded_time"] = data_trips_temp["start_time (local)"].dt.round("H")
    # Step 2: Combine rounded time and location for clustering
    data_trips_temp = data_trips_temp.dropna(subset=["rounded_time", "location[start][longitude]", "location[start][latitude]"])
    reference_timestamp = pd.Timestamp("1970-01-01", tz="UTC")
    data_trips_temp["rounded_time_unix"] = (data_trips_temp["rounded_time"] - reference_timestamp) // pd.Timedelta('1s')
    return data_trips_temp


def collecting_temp(data_trips_temp):
	# Apply Ridge Regression
	data_trips_temp = process_temperature_in_chunks(data_trips_temp, chunk_size=1)


	# Loop to repeat the function for 10,000 times
	for i in range(10000):
		print(f"Iteration {i + 1}/10000")

		# Call the function
		data_trips_temp = process_temperature_in_chunks(data_trips_temp, chunk_size=10)

		# Wait for 3 seconds before the next iteration
		time.sleep(0.1)


	data_trips_temp.loc[data_trips_temp["temperature"]==1000, "temperature"] = None
	data_trips_temp = data_trips_temp.sort_values(by=[ "temperature", "id"]).reset_index(drop=True)
	data_trips_temp.to_csv("data_trips_temp.csv", index=False)
	return data_trips_temp
# %%
# Calling the saved dataset


def calling_data():
	# Reading the input files
	codex_path = '/Users/haniftayarani/V2G_national/charging/codex.json'
	verbose = True
	data = process.load(codex_path)
	data_cahrging = data["charging"]
	data_trips = data["trips"]

	data_trips_temp = pd.read_csv('/Users/haniftayarani/V2G_national/charging/Data/data_trips_temp.csv')
	data_trips_temp = data_trips_temp[data_trips_temp["temperature"] < 1000]
	data_trips_temp = pd.merge(data_trips_temp,
							   data_trips[["id", "distance", "energy[consumption]"]],
							   how="left",
							   on="id")
	data_trips_temp = data_trips_temp[~data_trips_temp["energy[consumption]"].isna()]
	data_trips_temp = data_trips_temp[data_trips_temp["energy[consumption]"] < 0]
	data_trips_temp = data_trips_temp[data_trips_temp["distance"] > 0]
	data_trips_temp["energy_kwh/mi"] = abs(data_trips_temp["energy[consumption]"] / data_trips_temp["distance"])
	data_trips_temp = data_trips_temp[data_trips_temp["energy_kwh/mi"] < 1]
	data_trips_temp["temperature"] = data_trips_temp["temperature"].round(2)
	# Convert energy consumption from kWh/mile to J/meter
	data_trips_temp["energy_j/m"] = (data_trips_temp["energy_kwh/mi"] * 3.6e6) / 1609.34
	return data_trips_temp


def fit_with_ridge_with_equation(df, x_column, y_column, degree, alpha=1.0):
    """
    Fits a Ridge regression model with polynomial features and returns the polynomial function.

    Parameters:
    - df: DataFrame containing the data.
    - x_column: Column name for the x-axis.
    - y_column: Column name for the y-axis.
    - degree: Degree of the polynomial features.
    - alpha: Regularization strength.

    Returns:
    - poly_function: A callable function for the fitted polynomial.
    """
    x = df[[x_column]].values
    y = df[y_column].values

    # Create the model pipeline
    poly_features = PolynomialFeatures(degree)
    ridge_model = Ridge(alpha=alpha)
    model = make_pipeline(poly_features, ridge_model)
    model.fit(x, y)

    # Get the coefficients from the Ridge model
    coefficients = ridge_model.coef_
    intercept = ridge_model.intercept_

    # Generate the polynomial equation for display
    equation_terms = [
        f"{coeff:.3g}x^{i}" if i > 0 else f"{coeff:.3g}"
        for i, coeff in enumerate(coefficients)
    ]
    equation = " + ".join(equation_terms)
    equation = f"{intercept:.3g} + {equation}"
    print(f"Polynomial Equation (degree {degree}, alpha={alpha}): y = {equation}")

    # Create the callable polynomial function
    def poly_function(value):
        features = poly_features.fit_transform(np.array(value).reshape(-1, 1))
        return ridge_model.predict(features)

    # Generate fit values
    x_fit = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_fit = model.predict(x_fit)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, edgecolors='k', label="Data Points")
    plt.plot(x_fit, y_fit, color='red', label=f"y = {equation}")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Energy Consumption (J/m)")
    plt.title("Ridge Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

    return poly_function


def apply_polynomial_to_state_data(state_temp_data, polynomial_function, temperature_column="Celsius", new_column="Energy_Consumption"):
    """
    Applies a polynomial function to the temperature data in `state_temp_data` to calculate energy consumption.

    Parameters:
    - state_temp_data: DataFrame containing state temperature data.
    - polynomial_function: Polynomial function obtained from `fit_with_ridge_with_equation`.
    - temperature_column: The column name in `state_temp_data` for temperatures (default: "Value_Celsius").
    - new_column: The name of the new column to store energy consumption (default: "Energy_Consumption").

    Returns:
    - DataFrame: Updated `state_temp_data` with the new column for energy consumption.
    """
    # Ensure the temperature column exists
    if temperature_column not in state_temp_data.columns:
        raise ValueError(f"The temperature column '{temperature_column}' does not exist in the DataFrame.")

    # Apply the polynomial function to calculate energy consumption
    state_temp_data[new_column] = polynomial_function(state_temp_data[temperature_column])

    return state_temp_data


def state_mapping(df):

    state_mapping = {
        "Alabama": {"HHSTATE": "AL", "HHSTFIPS": 1},
        "Alaska": {"HHSTATE": "AK", "HHSTFIPS": 2},
        "Arizona": {"HHSTATE": "AZ", "HHSTFIPS": 4},
        "Arkansas": {"HHSTATE": "AR", "HHSTFIPS": 5},
        "California": {"HHSTATE": "CA", "HHSTFIPS": 6},
        "Colorado": {"HHSTATE": "CO", "HHSTFIPS": 8},
        "Connecticut": {"HHSTATE": "CT", "HHSTFIPS": 9},
        "Delaware": {"HHSTATE": "DE", "HHSTFIPS": 10},
        "Florida": {"HHSTATE": "FL", "HHSTFIPS": 12},
        "Georgia": {"HHSTATE": "GA", "HHSTFIPS": 13},
        "Hawaii": {"HHSTATE": "HI", "HHSTFIPS": 15},
        "Idaho": {"HHSTATE": "ID", "HHSTFIPS": 16},
        "Illinois": {"HHSTATE": "IL", "HHSTFIPS": 17},
        "Indiana": {"HHSTATE": "IN", "HHSTFIPS": 18},
        "Iowa": {"HHSTATE": "IA", "HHSTFIPS": 19},
        "Kansas": {"HHSTATE": "KS", "HHSTFIPS": 20},
        "Kentucky": {"HHSTATE": "KY", "HHSTFIPS": 21},
        "Louisiana": {"HHSTATE": "LA", "HHSTFIPS": 22},
        "Maine": {"HHSTATE": "ME", "HHSTFIPS": 23},
        "Maryland": {"HHSTATE": "MD", "HHSTFIPS": 24},
        "Massachusetts": {"HHSTATE": "MA", "HHSTFIPS": 25},
        "Michigan": {"HHSTATE": "MI", "HHSTFIPS": 26},
        "Minnesota": {"HHSTATE": "MN", "HHSTFIPS": 27},
        "Mississippi": {"HHSTATE": "MS", "HHSTFIPS": 28},
        "Missouri": {"HHSTATE": "MO", "HHSTFIPS": 29},
        "Montana": {"HHSTATE": "MT", "HHSTFIPS": 30},
        "Nebraska": {"HHSTATE": "NE", "HHSTFIPS": 31},
        "Nevada": {"HHSTATE": "NV", "HHSTFIPS": 32},
        "New Hampshire": {"HHSTATE": "NH", "HHSTFIPS": 33},
        "New Jersey": {"HHSTATE": "NJ", "HHSTFIPS": 34},
        "New Mexico": {"HHSTATE": "NM", "HHSTFIPS": 35},
        "New York": {"HHSTATE": "NY", "HHSTFIPS": 36},
        "North Carolina": {"HHSTATE": "NC", "HHSTFIPS": 37},
        "North Dakota": {"HHSTATE": "ND", "HHSTFIPS": 38},
        "Ohio": {"HHSTATE": "OH", "HHSTFIPS": 39},
        "Oklahoma": {"HHSTATE": "OK", "HHSTFIPS": 40},
        "Oregon": {"HHSTATE": "OR", "HHSTFIPS": 41},
        "Pennsylvania": {"HHSTATE": "PA", "HHSTFIPS": 42},
        "Rhode Island": {"HHSTATE": "RI", "HHSTFIPS": 44},
        "South Carolina": {"HHSTATE": "SC", "HHSTFIPS": 45},
        "South Dakota": {"HHSTATE": "SD", "HHSTFIPS": 46},
        "Tennessee": {"HHSTATE": "TN", "HHSTFIPS": 47},
        "Texas": {"HHSTATE": "TX", "HHSTFIPS": 48},
        "Utah": {"HHSTATE": "UT", "HHSTFIPS": 49},
        "Vermont": {"HHSTATE": "VT", "HHSTFIPS": 50},
        "Virginia": {"HHSTATE": "VA", "HHSTFIPS": 51},
        "Washington": {"HHSTATE": "WA", "HHSTFIPS": 53},
        "West Virginia": {"HHSTATE": "WV", "HHSTFIPS": 54},
        "Wisconsin": {"HHSTATE": "WI", "HHSTFIPS": 55},
        "Wyoming": {"HHSTATE": "WY", "HHSTFIPS": 56},
        "District of Columbia": {"HHSTATE": "DC", "HHSTFIPS": 11},
    }
    # Convert the mapping to a DataFrame
    state_mapping_df = pd.DataFrame.from_dict(state_mapping, orient="index").reset_index()
    state_mapping_df.columns = ["State", "HHSTATE", "HHSTFIPS"]

    # Merge with state_temp_data
    df = pd.merge(df, state_mapping_df, on="State", how="left")
    return df
# %%
#  State temp data

def fahrenheit_to_celsius():
    """
    Converts Fahrenheit temperatures in the 'Value' column of a CSV file to Celsius
    and calculates the average temperature per state.

    Returns:
        DataFrame: A DataFrame containing the state and the average temperature in Celsius.
    """
    # Read the CSV file
    temp = pd.read_csv("/Users/haniftayarani/V2G_national/charging/Data/Temp/temp.csv")

    # Ensure the necessary columns exist
    if "State" not in temp.columns or "Value" not in temp.columns:
        raise ValueError("The input file must contain 'State' and 'Value' columns.")

    # Convert Fahrenheit to Celsius
    temp["Celsius"] = (temp["Value"] - 32) * 5 / 9

    # Group by state and calculate the average Celsius temperature
    temp_avg = temp.groupby("State")["Celsius"].mean().reset_index()

    # Round the Celsius values to one decimal place
    temp_avg["Celsius"] = temp_avg["Celsius"].round(1)

    return temp_avg
#%%

def fit_with_poly():
    data_trips_temp = calling_data()
    df = data_trips_temp.copy()
    # Round temperatures to the nearest integer
    df['temperature_rounded'] = df['temperature'].round(0).astype(int)
    df = df[(df["temperature_rounded"]>0) & (df["temperature_rounded"]<46)]
    # Compute the mean energy consumption for each rounded temperature
    mean_data = df.groupby('temperature_rounded')['energy_j/m'].mean().reset_index()

    # Fit a polynomial (e.g., degree 2) to the mean data
    poly_degree = 2  # Change the degree as needed
    coeffs = np.polyfit(mean_data['temperature_rounded'], mean_data['energy_j/m'], poly_degree)
    poly_func = np.poly1d(coeffs)

    # Generate values for the polynomial curve
    temperature_range = np.linspace(df['temperature_rounded'].min(), df['temperature_rounded'].max(), 500)
    fitted_values = poly_func(temperature_range)

    # Create the box plot
    plt.figure(figsize=(12, 6))
    df.boxplot(column='energy_j/m', by='temperature_rounded', grid=False, showfliers=False)

    # Add the polynomial fit line
    plt.plot(temperature_range, fitted_values, color='red', label=f'Polynomial Fit (Degree {poly_degree})')

    # Customize the plot
    plt.title('Energy Consumption by Rounded Temperature with Polynomial Fit')
    plt.suptitle('')  # Remove automatic subtitle
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Energy Consumption')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()


def final_temp_adjustment():

    data_trips_temp = calling_data()
    polynomial_function = fit_with_ridge_with_equation(data_trips_temp, "temperature", "energy_kwh/mi", degree=3, alpha=0.5)
    fit_with_poly()
    state_temp_data = fahrenheit_to_celsius()
    # Apply the polynomial function to calculate energy consumption
    state_temp_data = apply_polynomial_to_state_data(state_temp_data, polynomial_function)
    state_temp_data = state_mapping(state_temp_data)
    return state_temp_data
