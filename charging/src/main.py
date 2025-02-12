# %%
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/Users/haniftayarani/V2G_national/charging/src')
from utilities import (nhts_state_count, generate_itineraries_for_selected_states,
                       run_itineraries_for_weeks, run_itineraries_for_weeks_delay,selecting_chuncks, aggregate_energy_demand, compute_pairwise_error_metrics, process_actual_charging, plot_charging_demand)
import process
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %% Input eVMT

# Reading the input files
codex_path = '/Users/haniftayarani/V2G_national/charging/codex.json'
verbose = True
data = process.load(codex_path)
data_charging = data["charging"]
data_trips = data["trips"]
nstate_counts, state_abbrev_to_name = nhts_state_count()
# emvt_charging_plot(data_charging)
# %% execute optimal charging for NHTS

# excluded_states = ["TX", "CA", "NY", "WI", "NC", "GA", "SC", "IA", "AZ", "MD", "FL"]  # Example states to exclude
# electricity_price_file = "/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json"
# # Generate itineraries for the remaining states
# selected_state_itineraries = generate_itineraries_for_selected_states(
#     day_types="all",
#     tile=3,
#     state_counts=nstate_counts,
#     excluded_states=excluded_states,
#     electricity_price_file=electricity_price_file
# )


# %% execute optimal charging for eVMT
##################################################################################################################################
# 3 Days chunk
##################################################################################################################################
results_dict_uncoordinated = run_itineraries_for_weeks(df_chagring=data_charging, max_days=365, chunk=3, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_3days.pkl", "wb") as f:
    pickle.dump(results_dict_uncoordinated, f)

results_dict_delay = run_itineraries_for_weeks_delay(df_chagring=data_charging, max_days=365, chunk=3, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_3days.pkl", "wb") as f:
    pickle.dump(results_dict_delay, f)
##################################################################################################################################
# 4 Days chunk
##################################################################################################################################
results_dict_uncoordinated = run_itineraries_for_weeks(df_chagring=data_charging, max_days=365, chunk=4, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_4days.pkl", "wb") as f:
    pickle.dump(results_dict_uncoordinated, f)

results_dict_delay = run_itineraries_for_weeks_delay(df_chagring=data_charging, max_days=365, chunk=4, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_4days.pkl", "wb") as f:
    pickle.dump(results_dict_delay, f)
##################################################################################################################################
# 5 Days chunk
##################################################################################################################################
results_dict_uncoordinated = run_itineraries_for_weeks(df_chagring=data_charging, max_days=365, chunk=5, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_5days.pkl", "wb") as f:
    pickle.dump(results_dict_uncoordinated, f)

results_dict_delay = run_itineraries_for_weeks_delay(df_chagring=data_charging, max_days=365, chunk=5, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_5days.pkl", "wb") as f:
    pickle.dump(results_dict_delay, f)
##################################################################################################################################
# 7 Days chunk
##################################################################################################################################
results_dict_uncoordinated = run_itineraries_for_weeks(df_chagring=data_charging, max_days=7, chunk=7, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_7days.pkl", "wb") as f:
    pickle.dump(results_dict_uncoordinated, f)

results_dict_delay = run_itineraries_for_weeks_delay(df_chagring=data_charging, max_days=365, chunk=7, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_7days.pkl", "wb") as f:
    pickle.dump(results_dict_delay, f)
##################################################################################################################################
# 14 Days chunk
##################################################################################################################################
results_dict_uncoordinated = run_itineraries_for_weeks(df_chagring=data_charging, max_days=365, chunk=14, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_14days.pkl", "wb") as f:
    pickle.dump(results_dict_uncoordinated, f)

results_dict_delay = run_itineraries_for_weeks_delay(df_chagring=data_charging, max_days=365, chunk=14, electricity_price_file="/Users/haniftayarani/V2G_national/charging/src/weighted_hourly_prices.json", state_name_abrv=state_abbrev_to_name)
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_14days.pkl", "wb") as f:
    pickle.dump(results_dict_delay, f)
# %%
with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_3days.pkl", "rb") as f:
    uncoordinated_results_3 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_3days.pkl", "rb") as f:
    delay_result_3 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_4days.pkl", "rb") as f:
    uncoordinated_results_4 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_4days.pkl", "rb") as f:
    delay_result_4 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_5days.pkl", "rb") as f:
    uncoordinated_results_5 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_5days.pkl", "rb") as f:
    delay_result_5 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_7days.pkl", "rb") as f:
    uncoordinated_results_7 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_7days.pkl", "rb") as f:
    delay_result_7 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_uncoordinated_365_14days.pkl", "rb") as f:
    uncoordinated_results_14 = pickle.load(f)

with open("/Users/haniftayarani/V2G_national/charging/Data/results_dict_delay_365_14days.pkl", "rb") as f:
    delay_result_14 = pickle.load(f)


test = delay_result_7[7]
test = uncoordinated_results_7[7]
# %%


def process_and_plot_heatmaps(uncoordinated_results, delay_result, chunk_duration):
    # Process charging data
    charging_uncoordinated = selecting_chuncks(uncoordinated_results)
    charging_delayed = selecting_chuncks(delay_result)

    # Aggregate energy demand
    energy_demand_uncoordinated = aggregate_energy_demand(charging_uncoordinated, interval=60)
    energy_demand_delayed = aggregate_energy_demand(charging_delayed, interval=60)

    # Compute error metrics
    error_matrices_uncoordinated = compute_pairwise_error_metrics(energy_demand_uncoordinated)
    error_matrices_delayed = compute_pairwise_error_metrics(energy_demand_delayed)

    # Function to plot and save heatmaps
    def plot_and_save_heatmap(matrix, metric_name, scenario, chunk_duration, title_font_size=20, axis_font_size=14, colorbar_font_size=12, axis_font_size_ticks=14):
        plt.figure(figsize=(24, 20))

        # Determine colorbar label based on the metric
        colorbar_label = {
            "MAE": "Mean Absolute Error (kWh)",
            "SMAPE (%)": "SMAPE (%)",
            "R-Squared": "R² (Coefficient of Determination)"
        }

        base_metric = metric_name.split("_")[0]
        cbar_label = colorbar_label.get(base_metric, "Value")

        # Plot the heatmap
        ax = sns.heatmap(matrix, cmap="coolwarm", linewidths=.5,
                         cbar_kws={"ticks": np.linspace(np.min(matrix), np.max(matrix), 10), "label": cbar_label})

        # Title and labels
        plt.title(f"{scenario} - {metric_name} Heatmap", fontsize=title_font_size)
        plt.xlabel("Duration of Study (Days)", fontsize=axis_font_size)
        plt.ylabel("Duration Chunk (Days)", fontsize=axis_font_size)

        # Adjust ticks
        plt.xticks(rotation=90, fontsize=axis_font_size_ticks)
        plt.yticks(rotation=0, fontsize=axis_font_size_ticks)

        # Adjust colorbar font size
        colorbar = ax.collections[0].colorbar
        colorbar.set_label(cbar_label, fontsize=colorbar_font_size + 4)
        colorbar.ax.tick_params(labelsize=colorbar_font_size)

        # Save plot
        filename = f"{scenario}_{metric_name.replace(' ', '_')}_{chunk_duration}days.png"
        plt.savefig(filename)
        plt.close()
        return filename

    # Metrics to plot
    metrics = ["MAE", "SMAPE (%)", "R-Squared"]

    # Plot and save heatmaps for both scenarios
    file_paths = []
    for scenario, error_matrices in zip(["Uncoordinated", "Delayed"], [error_matrices_uncoordinated, error_matrices_delayed]):
        for metric in metrics:
            matrix = error_matrices[metric]
            file_path = plot_and_save_heatmap(matrix, f"{metric}_365_{chunk_duration}days", scenario, chunk_duration,
                                              title_font_size=44, axis_font_size=36,
                                              colorbar_font_size=36, axis_font_size_ticks=22)
            file_paths.append(file_path)

    # Print saved file paths
    for path in file_paths:
        print(f"Saved heatmap: {path}")

    return energy_demand_uncoordinated, energy_demand_delayed


energy_demand_uncoordinated_7, energy_demand_delayed_7 = process_and_plot_heatmaps(uncoordinated_results_7, delay_result_7, 7)
energy_demand_uncoordinated_3, energy_demand_delayed_3 = process_and_plot_heatmaps(uncoordinated_results_3, delay_result_3, 3)
energy_demand_uncoordinated_14, energy_demand_delayed_14 = process_and_plot_heatmaps(uncoordinated_results_14, delay_result_14, 14)
# %%
# Example test:
# plot_charging_demand(energy_demand_uncordinated, actual_demand_df, num_scenarios=21)
# plot_charging_demand(energy_demand_delayed, actual_demand_df, num_scenarios=21)
#
# energy_demand_uncordinated[energy_demand_uncordinated["Scenario"] == 14].Energy_Demand_kWh.sum()
# uncordinated_results[14].Charging_kwh.sum() - uncordinated_results[7].Charging_kwh.sum()
#
# energy_demand_delayed[energy_demand_delayed["Scenario"] == 14].Energy_Demand_kWh.sum()
# delay_result[14].Charging_kwh.sum() - delay_result[7].Charging_kwh.sum()
#
# actual_demand_df.Energy_Demand_kWh.sum()
