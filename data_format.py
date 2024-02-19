import pandas as pd

# FILE PATHS:
input_csv_file_path = "Data/Wind_solar_demand_time_series.csv"
output_csv_file_path = "Data/averaged_values.csv"

# Load the data
data = pd.read_csv(input_csv_file_path)
print(data.info())

# take average of the data for each time
averaged_data_demand = data.groupby("Time", as_index=False)["Full demand"].mean()
averaged_data_wind = data.groupby("Time", as_index=False)["Wind CF"].mean()
averaged_data_solar = data.groupby("Time", as_index=False)["Solar CF"].mean()

# scale the data
averaged_data_demand["Full demand"] /= 30
averaged_data_wind["Wind CF"] /= 30
averaged_data_solar["Solar CF"] /= 20

# merge the data
averaged_data = pd.merge(averaged_data_demand, averaged_data_wind, on="Time")
averaged_data = pd.merge(averaged_data, averaged_data_solar, on="Time")

# save the data
averaged_data.to_csv(output_csv_file_path, index=False)