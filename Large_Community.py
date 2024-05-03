import time

import numpy as np

from Community_Model import CommunityModel
from Community_Utils import get_data, choose_generator, generation_data, load_data, \
    create_battery_matrix, compare_models_with_plot, display_bar_chart, show_saved_energy
from Env import T
from Probability_Distribution import get_probability_distribution

if __name__ == '__main__':
    # get train and test data with a split of 80%
    train_data, test_data = get_data(split=0.8)

    # get the probability distribution for the solar and wind data
    prob_solar = get_probability_distribution(train_data, "Solar")
    prob_wind = get_probability_distribution(train_data, "Wind")

    time_values = []
    mean_charging_1 = None
    mean_charging_2 = None
    savings_1 = None
    savings_2 = None

    i = 0
    while i <= 100:
        start = time.time()
        # get actual data for the simulation
        generators = choose_generator(num_agents=i)
        k = generation_data(test_data, generators, num_agents=i)
        h = load_data(test_data, num_agents=i)
        # battery characteristics
        s, c, d, e = create_battery_matrix(i)

        # define models
        community_model = CommunityModel(1, s, c, d, e, i, T)

        # run the simulation
        (mean_utility, mean_charging, savings, _,
         daily_utility, daily_charging, shapely_values, savings_2, exchange, mean_charging_2) = community_model.run_simulation(k, h, days=1)
        end = time.time()
        time_values.append((end - start) / 60 / 60)
        print("-----------------------------------")
        print(f"Time for {i} agents: {end - start}")
        print("-----------------------------------")
        print()
        i += 10

        mean_charging_1 = mean_charging
        savings_1 = savings

    # scale time values
    time_values = [x * 200 for x in time_values]

    # compare the results
    compare_models_with_plot(time_values, None, "Time", None, "Computation Time", "Number of Agents", "Time (s)",
                             file_name="Plots/time_comparison.png")

    # display the mean charging with bar
    display_bar_chart(mean_charging_1, "Mean Charging for 100 Agents", "Agents", "Mean Charging (kWh)",
                      file_name="Plots/mean_charging_comparison_large.png")

    # display the savings with bar
    show_saved_energy(savings_2[0:50], "Without Uncertainty", 50, file_name="Plots/saved_energy_2_large.png")


