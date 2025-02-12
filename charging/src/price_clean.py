import os
import pandas as pd
import json
import glob
import geopandas as gpd
# %%
class HourlyPriceProcessor:
    def __init__(self, input_directory, output_directory, combined_file, aggregated_file, county_shapefile, population_file, output_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.combined_file = combined_file
        self.aggregated_file = aggregated_file
        self.county_shapefile = county_shapefile
        self.population_file = population_file
        self.output_file = output_file
        self.aggregated_df = None
        os.makedirs(self.output_directory, exist_ok=True)  # Ensure output directory exists

    @staticmethod
    def safe_json_loads(val):
        """
        Safely parse JSON strings.
        """
        try:
            if isinstance(val, str):
                val = val.replace("'", "\"")
            return json.loads(val)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing JSON: {val} - {e}")
            return None

    def combine_csv_files(self):
        """
        Reads and combines all CSV files in the specified directory into a single CSV file,
        keeping only rows where the 'enddate' column is not null.
        """
        all_files = [
            os.path.join(self.input_directory, f)
            for f in os.listdir(self.input_directory)
            if f.endswith('.csv')
        ]

        if not all_files:
            print("No CSV files found in the directory.")
            return

        combined_df = pd.DataFrame()

        for file in all_files:
            try:
                print(f"Reading file: {file}")
                df = pd.read_csv(file)
                if 'enddate' in df.columns:
                    df = df[df['enddate'].isna()]
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading file {file}: {e}")

        combined_df.to_csv(self.combined_file, index=False)
        print(f"Combined data saved to {self.combined_file}")

    @staticmethod
    def process_energyratestructure(energyratestructure):
        """
        Simplify the energyratestructure to keep only the last dictionary of each tier.
        """
        if not energyratestructure or not isinstance(energyratestructure, list):
            return []
        simplified_structure = []
        for tier in energyratestructure:
            if isinstance(tier, list) and len(tier) > 0:
                last_dict = tier[-1]
                if isinstance(last_dict, dict):
                    simplified_structure.append([last_dict])
        return simplified_structure

    def preprocess_data(self):
        """
        Load and preprocess the combined CSV file.
        """
        test = pd.read_csv(self.combined_file)
        test['energyratestructure'] = test['energyratestructure'].apply(self.safe_json_loads)
        test['energyratestructure'] = test['energyratestructure'].apply(self.process_energyratestructure)
        test['energyweekdayschedule'] = test['energyweekdayschedule'].apply(self.safe_json_loads)
        test['energyweekendschedule'] = test['energyweekendschedule'].apply(self.safe_json_loads)
        return test

    @staticmethod
    def calculate_hourly_prices(row, hourly_df):
        try:
            weekday_schedule = row['energyweekdayschedule']
            weekend_schedule = row['energyweekendschedule']
            ratestructure = row['energyratestructure']

            if not (isinstance(weekday_schedule, list) and len(weekday_schedule) == 12):
                print(f"Invalid weekday schedule for GEOID: {row['GEOID']}")
                return None
            if not (isinstance(weekend_schedule, list) and len(weekend_schedule) == 12):
                print(f"Invalid weekend schedule for GEOID: {row['GEOID']}")
                return None
            if not (isinstance(ratestructure, list) and len(ratestructure) > 0):
                print(f"Invalid ratestructure for GEOID: {row['GEOID']}")
                return None

            tier_rates = {tier_index: float(rate[0]['rate']) for tier_index, rate in enumerate(ratestructure)}

            def get_price(is_weekend, month, hour_of_day):
                schedule = weekend_schedule if is_weekend else weekday_schedule
                tier = schedule[month][hour_of_day]
                return tier_rates.get(tier, 0.0)

            hourly_df['price'] = hourly_df.apply(
                lambda x: get_price(x['is_weekend'], x['month'], x['hour_of_day']),
                axis=1
            )

            hourly_prices = hourly_df['price'].values.flatten()
            result_row = {
                'GEOID': row['GEOID'],
                'lat': row['lat'],
                'lon': row['lon'],
                'sector': row['sector'],
                'utility_name': row['utility_name'],
                'Rate_name': row['Rate_name'],
                **{f'hour_{i}': price for i, price in enumerate(hourly_prices)}
            }
            return result_row
        except Exception as e:
            print(f"Error processing GEOID {row['GEOID']}: {e}")
            return None

    def process_and_save_chunks(self, test, hourly_df, chunk_size=1000):
        """
        Process data in chunks and save each chunk as a separate CSV file.
        """
        for chunk_index, start_idx in enumerate(range(0, len(test), chunk_size)):
            chunk = test.iloc[start_idx:start_idx + chunk_size]
            chunk_rows = []

            for _, row in chunk.iterrows():
                if row['energyratestructure'] is None or row['energyweekdayschedule'] is None or row['energyweekendschedule'] is None:
                    print(f"Skipping row with invalid data: {row['GEOID']}")
                    continue

                hourly_data = self.calculate_hourly_prices(row, hourly_df.copy())
                if hourly_data is not None:
                    chunk_rows.append(hourly_data)

            if chunk_rows:
                chunk_df = pd.DataFrame(chunk_rows)
                output_file = os.path.join(self.output_directory, f"hourly_prices_chunk_{chunk_index + 1}.csv")
                chunk_df.to_csv(output_file, index=False)
                print(f"Saved {len(chunk_df)} rows to {output_file}")

    def aggregate_data(self):
        """
        Combine all chunk files, group by GEOID and sector, and calculate average hourly prices.
        """
        all_files = glob.glob(os.path.join(self.output_directory, "*.csv"))
        df_list = []

        for file in all_files:
            print(f"Loading {file}")
            df = pd.read_csv(file)
            df_list.append(df)

        combined_df = pd.concat(df_list, ignore_index=True)
        print("All files combined.")

        columns_to_drop = ['lat', 'lon', 'utility_name', 'Rate_name']
        if any(col in combined_df.columns for col in columns_to_drop):
            combined_df.drop(columns=columns_to_drop, inplace=True)
            print(f"Columns dropped: {columns_to_drop}")

        hour_columns = [col for col in combined_df.columns if col.startswith("hour_")]
        self.aggregated_df = combined_df.groupby(['GEOID', 'sector'])[hour_columns].mean().reset_index()
        self.aggregated_df.to_csv(self.aggregated_file, index=False)
        print(f"Aggregated data saved to {self.aggregated_file}")

    def weighted_average(self):
        self.aggregated_df = pd.read_csv("Data/aggregated_hourly_prices.csv")
        # Load GEOID to county/state mapping
        self.gdf_county = gpd.read_file(county_shapefile)
        self.county_data = self.gdf_county[['GEOID', 'NAME', "STATE_NAME"]].rename(columns={'NAME': 'county', "STATE_NAME": "state"})
        self.county_data['GEOID'] = self.county_data['GEOID'].astype(int)
        self.aggregated_df['GEOID'] = self.aggregated_df['GEOID'].astype(int)

        # Extract county and state abbreviation
        # Load population data
        self.population_data = pd.read_csv(population_file)
        self.population_data = self.population_data.drop(columns=["county_ascii", "county_full", "state_id", "lat", "lng"], errors='ignore')

        # Merge county/state and population data
        self.merged_data = pd.merge(self.county_data, self.population_data, left_on=["county", "state"], right_on=["county", "state_name"], how="left")
        self.merged_data = self.merged_data.dropna()

        # Calculate the total population for each state
        self.merged_data['State_Total_Population'] = self.merged_data.groupby('state')['population'].transform('sum')

        # Calculate the weight of each county's population within its state
        self.merged_data['Population_Weight'] = self.merged_data['population'] / self.merged_data['State_Total_Population']

        self.weight_price = pd.merge(self.merged_data[["GEOID", "county", "state", "population", "Population_Weight"]], self.aggregated_df, on="GEOID", how="left")
        self.weight_price = self.weight_price.dropna()

        # Identify columns that start with "hour_"
        hour_columns = [col for col in self.weight_price.columns if col.startswith("hour_")]

        # Group by state and sector, and compute weighted averages
        self.weighted_avg = self.weight_price.groupby(['state', 'sector']).apply(
            lambda group: pd.Series({
                col: (group[col] * group['Population_Weight']).sum() / group['Population_Weight'].sum()
                for col in hour_columns
            })
        ).reset_index()

        # Rename columns for clarity
        self.weighted_avg.columns = ['state', 'sector'] + hour_columns
        # Convert to nested dictionary format
        nested_dict = {}
        for _, row in self.weighted_avg.iterrows():
            state = row['state']
            sector = row['sector']
            rates = {i: round(row[f"hour_{i}"], 3) for i in range(len(hour_columns))}

            if state not in nested_dict:
                nested_dict[state] = {}
            if sector not in nested_dict[state]:
                nested_dict[state][sector] = {'rate': rates}

        # Save the nested dictionary to a JSON file
        output_json_file = "weighted_hourly_prices.json"
        with open(output_json_file, "w") as json_file:
            json.dump(nested_dict, json_file, indent=4)

# %%
if __name__ == "__main__":
    input_directory = "Data/Price_data"
    output_directory = "hourly_price_chunks"
    combined_file = "Data/combined_price_data.csv"
    aggregated_file = "Data/aggregated_hourly_prices.csv"
    county_shapefile = "/Users/haniftayarani/V2G_national/charging/Data/census/cb_2023_us_county_5m.shp"
    population_file = "/Users/haniftayarani/V2G_national/charging/Data/population/uscounties.csv"
    output_file = "Data/weighted_hourly_prices.csv"
    processor = HourlyPriceProcessor(input_directory, output_directory, combined_file, aggregated_file, county_shapefile, population_file, output_file)

    # # Step 1: Combine CSV files
    # processor.combine_csv_files()
    #
    # # Step 2: Preprocess data
    # test = processor.preprocess_data()
    #
    # # Step 3: Create hourly timeframe
    # hourly_timeframe = pd.date_range(start="2024-01-01", end="2024-12-31 23:00", freq="H")
    # hourly_df = pd.DataFrame({'timestamp': hourly_timeframe})
    # hourly_df['month'] = hourly_df['timestamp'].dt.month - 1
    # hourly_df['day_of_week'] = hourly_df['timestamp'].dt.dayofweek
    # hourly_df['hour_of_day'] = hourly_df['timestamp'].dt.hour
    # hourly_df['is_weekend'] = hourly_df['day_of_week'].isin([5, 6])
    #
    # # Step 4: Process and save in chunks
    # processor.process_and_save_chunks(test, hourly_df)
    #
    # # Step 5: Aggregate data
    # processor.aggregate_data()
    processor.weighted_average()
