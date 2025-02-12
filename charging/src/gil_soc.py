# %%
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/Users/haniftayarani/V2G_national/charging/src')
import pandas as pd
from utilities import load_itineraries, read_pkl
import optimization
import process
from reload import deep_reload
import numpy as np
from itineraries import MakeItineraries
from adoption import combine_csv_files
import time
import matplotlib.pyplot as plt
import seaborn as sns

# %%


def plot_losses_per_hour_boxplot(data, x_column,x_label, font_size=14):
    """
    Create a box plot showing energy losses per hour of parking by vehicle model.

    Parameters:
        data (DataFrame): The input DataFrame containing vehicle loss data.
        font_size (int, optional): Size of the fonts for titles, labels, and ticks.
    """
    # Create the box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data, x=x_column, y='losses_hour_parking', palette='Set2')

    # Customize plot
    plt.title(f'Energy Losses by {x_label}', fontsize=font_size + 2)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel('Losses (kWh per Hour of Parking)', fontsize=font_size)
    plt.xticks(rotation=45, fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()

def plot_total_loss_by_parking_duration(data, font_size=14):
    """
    Plot total energy losses by parking duration (in days) for each vehicle model.

    Parameters:
        data (DataFrame): The input DataFrame containing parking session data.
        font_size (int, optional): Size of the fonts for titles, labels, and legend.
    """
    # Convert parking duration from minutes to days
    data['parking_duration_days'] = data['parking_duration_minutes'] / (60 * 24)

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=data,
        x='parking_duration_days',
        y='losses_kWh',
        hue='vehicle_model',
        s = 200,
        marker='o',
    )

    # Customize plot
    plt.title('Total Energy Losses vs. Total Parking Duration Over a Year by Vehicle Model', fontsize=font_size + 2)
    plt.xlabel('Total Parking Duration (Days)', fontsize=font_size)
    plt.ylabel('Total Energy Losses (kWh)', fontsize=font_size)
    plt.legend(title='Vehicle Model', fontsize=font_size - 2, title_fontsize=font_size)
    plt.grid(True)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.ylim(0,600)

    # Show plot
    plt.tight_layout()
    plt.show()

def plot_total_loss_with_trendlines(data, font_size=14):
    """
    Plot total energy losses by parking duration (in days) with trendlines and confidence intervals for each vehicle model.

    Parameters:
        data (DataFrame): The input DataFrame containing parking session data.
        font_size (int, optional): Size of the fonts for titles, labels, and legend.
    """
    # Convert parking duration from minutes to days
    data['parking_duration_days'] = data['parking_duration_minutes'] / (60)

    # Create the lmplot
    g = sns.lmplot(
        data=data,
        x='parking_duration_days',
        y='losses_kWh',
        hue='vehicle_model',
        scatter_kws={'s': 100, 'alpha': 0.7},
        line_kws={'linewidth': 2},
        height=8,
        aspect=1.5,
        ci=95  # Adds the shaded confidence interval around the line
    )

    # Remove the automatic legend created by Seaborn
    g._legend.remove()

    # Create a custom legend
    ax = g.ax
    handles, labels = ax.get_legend_handles_labels()  # Get colors from the plot
    plt.legend(
        handles=handles,
        labels=labels,
        title='Vehicle Model',
        loc='upper right',
        fontsize=font_size - 2,
        title_fontsize=font_size,
        frameon=True  # Add a border around the legend
    )

    # Customize plot
    g.set_axis_labels('Total Parking Duration (Hour)', 'Total Energy Losses (kWh)', fontsize=font_size)
    g.fig.suptitle('Total Energy Losses vs. Total Parking Duration Over a Year by Vehicle Model', fontsize=font_size + 2)
    plt.ylim(0,600)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_duration_vs_losses(data, group_by_column, max_duration=None, font_size=14):
    """
    Create a grid of scatter plots showing parking duration vs. energy losses grouped by a specified column.

    Parameters:
        data (DataFrame): The input DataFrame containing parking session data.
        group_by_column (str): Column to group the data by (e.g., 'season', 'destination').
        max_duration (int, optional): Maximum duration for x-axis. If None, use the data's max duration.
        font_size (int, optional): Size of the fonts for titles, labels, and legend.
    """
    # Filter data for the maximum duration if provided
    if max_duration:
        data = data[data['parking_duration_minutes'] <= max_duration]

    # Get unique values in the grouping column
    unique_values = data[group_by_column].unique()

    # Replace underscores in labels for better readability
    unique_values = [str(value).replace('_', ' ') for value in unique_values]

    # Determine grid size based on the number of unique values
    n_plots = len(unique_values)
    n_rows = (n_plots // 2) + (n_plots % 2)
    n_cols = 2 if n_plots > 1 else 1

    # Create the grid of plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 6), sharex=True, sharey=True)
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Plot each group
    for i, value in enumerate(unique_values):
        ax = axes[i]
        # Filter data for the current group
        group_data = data[data[group_by_column].replace('_', ' ') == value]
        # Create a scatter plot
        sns.scatterplot(
            data=group_data,
            x='parking_duration_minutes',
            y='losses_kWh',
            hue='vehicle_model',
            alpha=0.7,
            s=100,  # Increase point size
            ax=ax
        )
        formatted_column_name = group_by_column.replace('_', ' ').capitalize()
        ax.set_title(f'{formatted_column_name}: {value}', fontsize=font_size + 2)
        ax.set_xlabel('Parking Duration (minutes)', fontsize=font_size)
        ax.set_ylabel('Energy Losses (kWh)', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=font_size - 2, title='Vehicle Model', title_fontsize=font_size)

    # Ensure all subplots share the same x-axis ticks for readability
    for ax in axes:
        ax.tick_params(labelbottom=True)

    # Hide unused subplots if the grid has extra slots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

def get_season(month, day):
    if (month == 12 and day >= 21) or (month in [1, 2]) or (month == 3 and day < 21):
        return "Winter"
    elif (month == 3 and day >= 21) or (month in [4, 5]) or (month == 6 and day < 21):
        return "Spring"
    elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day < 21):
        return "Summer"
    elif (month == 9 and day >= 21) or (month in [10, 11]) or (month == 12 and day < 21):
        return "Fall"


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
                return min(capacities)
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
                return min(capacities)
    # Return NaN if no match
    return np.nan

def draw_pie_chart_with_nan_as_parking_and_legend(data, column, font_size=14, title="Pie Chart"):
    """
    Draw a pie chart based on the specified column in the dataset, treating NaN values as "Parking" and adding a legend.

    Parameters:
        data (DataFrame): The input DataFrame containing the data.
        column (str): The column to use for grouping the pie chart.
        font_size (int, optional): Font size for the title and labels. Default is 14.
        title (str, optional): Title of the pie chart. Default is "Pie Chart".
    """
    # Fill NaN values with "Parking"
    data[column] = data[column].fillna("Parking")

    # Calculate the value counts for the chosen column
    counts = data[column].value_counts()

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,  # Exclude labels from slices
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': font_size}
    )
    for autotext in autotexts:
        autotext.set_fontsize(font_size + 1)
    # Add a legend
    ax.legend(wedges, counts.index, loc='center left', bbox_to_anchor=(0.1, 0.1), fontsize=font_size+4, title_fontsize=font_size+2, title="Event")

    # Add title
    plt.title(title, fontsize=font_size + 2)

    plt.tight_layout()
    plt.show()


def get_lengths_dataframe(df1, df2, column_names=('Dataset 1', 'Dataset 2')):
    # Calculate the lengths of the DataFrames
    lengths = {column_names[0]: len(df1), column_names[1]: len(df2)}

    # Convert to a DataFrame
    lengths_df = pd.DataFrame(list(lengths.items()), columns=['Dataset', 'Length'])

    return lengths_df

def draw_pie_chart_with_legend(lengths_df, font_size=14, title="Pie Chart of Dataset Lengths"):
    """
    Draw a pie chart from a DataFrame containing dataset lengths with percentages and labels in a legend.

    Parameters:
        lengths_df (DataFrame): A DataFrame with columns ['Dataset', 'Length'].
        font_size (int, optional): Font size for the title and labels. Default is 14.
        title (str, optional): Title of the pie chart. Default is "Pie Chart of Dataset Lengths".
    """
    # Ensure the DataFrame has the correct columns
    if not {'Dataset', 'Length'}.issubset(lengths_df.columns):
        raise ValueError("Input DataFrame must contain 'Dataset' and 'Length' columns.")

    # Plot the pie chart with percentages
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        lengths_df['Length'],
        labels=None,  # Remove labels from the pie slices
        autopct='%1.1f%%',  # Display percentages on the pie slices
        startangle=270,
        colors=plt.cm.Set2.colors[:len(lengths_df)],

    )

    # Customize percentages (autotexts)
    for autotext in autotexts:
        autotext.set_fontsize(font_size + 2)

    # Add a legend
    ax.legend(
        wedges,
        lengths_df['Dataset'],
        title="Dataset",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),  # Place legend below the chart
        fontsize=font_size + 2,
        title_fontsize=font_size
    )

    # Add title
    plt.title(title, fontsize=font_size + 2)
    plt.tight_layout()
    plt.show()

# %%

trips = pd.read_csv("/Users/haniftayarani/V2G_national/charging/Data/Raw/bev_trips_full.csv")
charging = pd.read_csv("/Users/haniftayarani/V2G_national/charging/Data/Raw/bev_zcharges.csv")
temp = pd.read_csv("/Users/haniftayarani/V2G_national/charging/Data/data_trips_temp.csv")
trips = pd.merge(trips, charging[["last_trip_id", "energy[charge_type][type]"]], right_on="last_trip_id", left_on="id", how="left")
trips = trips[~trips["battery[soc][start]"].isna()]
trips = pd.merge(trips, temp[["id", "temperature"]], on="id", how="left")
trips = trips.sort_values(by=["vehicle_name", "start_time (local)"])
trips['SOC_at_end_of_parking'] = trips.groupby('vehicle_name')['battery[soc][start]'].shift(-1)
trips['next_start_time'] = trips.groupby('vehicle_name')['start_time (local)'].shift(-1)
trips['SOC_at_start_of_parking'] = trips['battery[soc][end]']
trips['temp_at_start_of_parking'] = trips['temperature']
trips['temp_at_end_of_parking'] = trips.groupby('vehicle_name')['temperature'].shift(-1)
trips['average_temp_of_parking'] =(trips['temp_at_start_of_parking'] + trips['temp_at_end_of_parking'])/2
trips = trips[~trips["SOC_at_end_of_parking"].isna()]
parking_no_charging = trips[trips["energy[charge_type][type]"].isna()]
parking_no_charging = parking_no_charging[parking_no_charging["charge_level"].isna()]
parking_no_charging["SOC_at_end_of_parking"] = parking_no_charging["SOC_at_end_of_parking"].round(1)
parking_no_charging["SOC_at_start_of_parking"] = parking_no_charging["SOC_at_start_of_parking"].round(1)


parking_no_charging_nochange = parking_no_charging[(parking_no_charging["SOC_at_end_of_parking"]) == (parking_no_charging["SOC_at_start_of_parking"])]
parking_no_charging = parking_no_charging[(parking_no_charging["SOC_at_end_of_parking"]) <= (parking_no_charging["SOC_at_start_of_parking"])]

# Convert time columns to datetime
parking_no_charging["next_start_time"] = pd.to_datetime(parking_no_charging["next_start_time"], format='mixed')
parking_no_charging["start_time (local)"] = pd.to_datetime(parking_no_charging["start_time (local)"], format='mixed')
parking_no_charging["parking_duration"] = parking_no_charging["next_start_time"] - parking_no_charging["start_time (local)"]
parking_no_charging["parking_duration"] = pd.to_timedelta(parking_no_charging["parking_duration"])
parking_no_charging["parking_duration_minutes"] = parking_no_charging["parking_duration"].dt.total_seconds() /60
# Apply the function to create the season column
parking_no_charging['season'] = parking_no_charging.apply(lambda row: get_season(row['month'], row['day']), axis=1)
# Apply the function to create the battery_capacity column
parking_no_charging['battery_capacity_kwh'] = parking_no_charging.apply(get_battery_capacity, axis=1)
# Calculate SOC difference as a percentage
parking_no_charging['SOC_difference'] = parking_no_charging['SOC_at_start_of_parking'] - parking_no_charging['SOC_at_end_of_parking']
parking_no_charging = parking_no_charging[parking_no_charging["parking_duration_minutes"] > 0]
# Calculate losses in kWh
parking_no_charging['losses_kWh'] = (parking_no_charging['SOC_difference'] / 100) * parking_no_charging['battery_capacity_kwh']
parking_no_charging['losses_kWh_hour'] = parking_no_charging['losses_kWh'] / (parking_no_charging['parking_duration_minutes']/60)
vehicle_loss = parking_no_charging.groupby(["vehicle_name", "vehicle_model", "battery_capacity_kwh"])[["parking_duration_minutes", "losses_kWh"]].sum().reset_index(drop=False)
vehicle_loss["losses_hour_parking"] = vehicle_loss["losses_kWh"] / (vehicle_loss["parking_duration_minutes"]/60)


parking_no_charging["loss_during_parking"] = False
parking_no_charging.loc[
    parking_no_charging["SOC_at_end_of_parking"] < parking_no_charging["SOC_at_start_of_parking"],
    "loss_during_parking"
] = True
# %%
plot_duration_vs_losses(parking_no_charging, group_by_column='season', max_duration=20000, font_size=16)
plot_duration_vs_losses(parking_no_charging, group_by_column='destination_label', max_duration=20000, font_size=16)

plot_losses_per_hour_boxplot(vehicle_loss,'vehicle_model',"Vehicle Model",font_size=16)
plot_losses_per_hour_boxplot(vehicle_loss,'battery_capacity_kwh',"Battery Capacity (kWh)", font_size=16)
plot_total_loss_by_parking_duration(vehicle_loss, font_size=16)
plot_total_loss_with_trendlines(vehicle_loss, font_size=16)

# %%

# Create temperature bins
bins = [-20, 0, 10, 20, 30, 40]
labels = ['≤0°C', '0-10°C', '10-20°C', '20-30°C', '30-40°C']
parking_no_charging['temp_range'] = pd.cut(parking_no_charging['average_temp_of_parking'], bins=bins, labels=labels)

# Boxplot: Losses by temperature range
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=parking_no_charging,
    x='temp_range',
    y='losses_kWh_hour',
    hue='vehicle_model',
)
plt.title('Losses by Temperature Range', fontsize=16)
plt.xlabel('Temperature Range (°C)', fontsize=14)
plt.ylabel('Losses (kWh/hour)', fontsize=14)
plt.legend(title='Vehicle Model', fontsize=12, title_fontsize=14)
plt.grid(True, axis='y')
plt.ylim(0,0.5)
plt.tight_layout()
plt.show()
# %%
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
parking_no_charging['soc_range'] = pd.cut(parking_no_charging['SOC_at_start_of_parking'], bins=bins, labels=labels)

# Boxplot: Losses by temperature range
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=parking_no_charging,
    x='soc_range',
    y='losses_kWh_hour',
    hue='vehicle_model',
)
plt.title('', fontsize=16)
plt.xlabel('SOC at start of parking', fontsize=14)
plt.ylabel('Losses (kWh/hour)', fontsize=14)
plt.legend(title='Vehicle Model', fontsize=12, title_fontsize=14)
plt.grid(True, axis='y')
plt.ylim(0,1)
plt.tight_layout()
plt.show()

# %%
draw_pie_chart_with_nan_as_parking_and_legend(trips, column='energy[charge_type][type]', font_size=16, title="Distribution of Charging Behavior")
draw_pie_chart_with_nan_as_parking_and_legend(parking_no_charging, column='vehicle_model', font_size=14, title="")
length = get_lengths_dataframe(parking_no_charging, parking_no_charging_nochange, column_names=('Parking with losses', 'Parking without losses'))
draw_pie_chart_with_legend(length, font_size=18, title="")
