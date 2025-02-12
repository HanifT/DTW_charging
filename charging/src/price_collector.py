import requests
import geopandas as gpd
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
# %%

class ElectricityDataCollector:
    def __init__(self, api_key, base_url, geometry_url, output_dir="Data/ACS_2021/Tract_Geometries"):
        self.api_key = api_key
        self.base_url = base_url
        self.geometry_url = geometry_url
        self.output_dir = output_dir
        self.consolidated_data = pd.DataFrame()

    def download_and_extract_geometry(self):
        print("Downloading Census Geometry Data...")
        response = requests.get(self.geometry_url)
        response.raise_for_status()

        with ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(self.output_dir)
        print("Geometry Data Extracted")

    def load_geometry(self, shapefile_name):
        filepath = f"{self.output_dir}/{shapefile_name}"
        tracts = gpd.read_file(filepath)
        tracts['lon'] = tracts.geometry.centroid.x
        tracts['lat'] = tracts.geometry.centroid.y
        return pd.DataFrame(tracts[['GEOID', 'lon', 'lat']])

    def fetch_utility_data(self, lat, lon, sector=["Commercial"]):
        params = {
            'api_key': self.api_key,
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'sector': sector,
            'detail': 'full'
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data for lat: {lat}, lon: {lon}, sector: {sector}")
            return None

    def process_and_append_data(self, geoid, lat, lon, items, sector):
        for item in items:
            new_row = {
                'GEOID': geoid,
                'lat': lat,
                'lon': lon,
                'sector': sector,
                'utility_name': item.get('utility', 'N/A'),
                'Rate_name': item.get('name', 'N/A'),
                'startdate': item.get('startdate', 'N/A'),
                'enddate': item.get('enddate', 'N/A'),
                'demandrate': item.get('demandratestructure', []),
                'demandweekdayschedule': item.get('demandweekdayschedule', []),
                'demandweekendschedule': item.get('demandweekendschedule', []),
                'energyratestructure': item.get('energyratestructure', []),
                'energyweekdayschedule': item.get('energyweekdayschedule', []),
                'energyweekendschedule': item.get('energyweekendschedule', []),
                'fixedmonthlycharge': item.get('fixedmonthlycharge', 'N/A'),
                'minmonthlycharge': item.get('minmonthlycharge', 'N/A'),
                'annualmincharge': item.get('annualmincharge', 'N/A'),
                'fixedattrs': item.get('fixedattrs', 'N/A')
            }
            self.consolidated_data = pd.concat([self.consolidated_data, pd.DataFrame([new_row])], ignore_index=True)

    def collect_data(self, geometry_df, start_index=0, end_index=None, sectors=['Commercial']):
        subset_df = geometry_df.iloc[start_index:end_index]

        for index, row in subset_df.iterrows():
            print(f"Processing index: {index}")
            for sector in sectors:
                data = self.fetch_utility_data(row['lat'], row['lon'], sector)

                if data and 'items' in data:
                    self.process_and_append_data(row['GEOID'], row['lat'], row['lon'], data['items'], sector)
                else:
                    print(f"No 'items' key for GEOID: {row['GEOID']} (Sector: {sector})")

    def save_data(self, output_file):
        self.consolidated_data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

# %%


# Example API:
# 9ED4OxQpOHLCajotcjhqAvvN1BebbVG4dVk7AcyU (ucdavis.edu)
# oPuhNmPIkLWHqF0ZyVyjCIdjdb4qABKIRxFGm3p4 (gmail.com)
# sf8e7F2VAnPlrubWaXmKXJRxYMRsBXbU2OUF4ONm (gmail2.com)
# YK13Yb1bbOVF2upGFl6R4LZAorqtP2HEy9Esnkcx (ymail.com)
# dMqiJxKJKsureHF0Yjl05sln5jV5A1bJBOT4ApvO (icloud.com)
# 9pgDECAfK84F15lhoibVukiA4tpFg9qW4yiJdd0J (bardia.com)
# HQxSav0P2JygohgVM2UyznH5SfBzOTuacQw6OEif
# tHZ2quQ0iVJlrig94UYVvfjTA4oGWLh5lQwXOwQ2
# jc5TrK7j1w90Nf86Su3oCZZJrK0Vw9vXdOvhbuDh
# b9wvAGJtBlffcBbsUahIk7ydtF6049rUGUWXJbGw


if __name__ == "__main__":
    API_KEY = 'tHZ2quQ0iVJlrig94UYVvfjTA4oGWLh5lQwXOwQ2'
    BASE_URL = 'https://api.openei.org/utility_rates?version=3&'
    GEOMETRY_URL = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_5m.zip"
    SHAPEFILE_NAME = "cb_2021_us_county_5m.shp"

    collector = ElectricityDataCollector(API_KEY, BASE_URL, GEOMETRY_URL)

    # Step 1: Download and extract geometry data
    collector.download_and_extract_geometry()

    # Step 2: Load the geometry data
    geometry_df = collector.load_geometry(SHAPEFILE_NAME)

    # Step 3: Collect utility data for a subset of geometry locations
    collector.collect_data(geometry_df, start_index=3014, end_index=3233, sectors=['Commercial', 'Residential'])

    # Step 4: Save the collected data to a CSV file
    collector.save_data("consolidated_data.csv")
