import pandas as pd
from matplotlib import pyplot as plt

# FILE PATHS:
input_csv_file_path = "Data/.csv"
output_dir_path = "Data/"

# Load the data
data = pd.read_csv(input_csv_file_path)
print(data.info())


def get_day_from_data(input_data, date):
    return input_data[input_data["% Date"] == date]


def plot_daily_data(input_day, date):
    plt.plot(input_day["Time"], input_day["Full demand"], label="Load", color="green", marker="o")
    plt.plot(input_day["Time"], input_day["Solar CF"], label="Solar", color="red", marker="o")
    plt.plot(input_day["Time"], input_day["Wind CF"], label="Wind", color="blue", marker="o")
    # only display every 4th x-label
    plt.xticks(input_day["Time"][::4])
    plt.xlabel("Time")
    plt.ylabel("kWh")
    plt.title(f"Daily data for {date}")
    plt.legend()
    plt.show()


day = get_day_from_data(data, "03.10.2014")
print(day)

plot_daily_data(day, "01.01.2014")
