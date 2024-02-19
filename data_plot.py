import pandas as pd
import matplotlib.pyplot as plt

# FILE PATHS:
input_csv_file_path = "Data/averaged_values.csv"
output_png_file_path = "Data/averaged_values_plot.png"

# Load the data
averaged_data = pd.read_csv(input_csv_file_path)

# scale the data
averaged_data["Wind CF"] /= 1.5

# plot the data
plt.plot(averaged_data["Time"], averaged_data["Full demand"], label="Demand", color="green", linestyle="-", marker="o")
plt.plot(averaged_data["Time"], averaged_data["Wind CF"], label="Wind", color="blue", linestyle="-", marker="o")
plt.plot(averaged_data["Time"], averaged_data["Solar CF"], label="Solar", color="red", linestyle="-", marker="o")
plt.xticks(averaged_data["Time"][::4])
plt.xlabel("Time")
plt.ylabel("KWh")
plt.title("Averaged Values for a Day")
plt.legend()
# save the plot
plt.savefig(output_png_file_path)
print(F"Plot saved to {output_png_file_path}")
# show the plot
plt.show()


