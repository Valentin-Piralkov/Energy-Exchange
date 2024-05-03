import numpy as np

from Community_Model import CommunityModel
from Community_Utils import compare_utility, get_data, choose_generator, generation_data, load_data, compare_charging, \
    create_battery_matrix, compare_models_with_plot, compare_wasted_energy, show_saved_energy
from Env import NUM_AGENT, T, DAYS
from Probability_Distribution import get_probability_distribution

if __name__ == '__main__':
    # get train and test data with a split of 80%
    train_data, test_data = get_data(split=0.8)

    # get actual data for the simulation
    generators = choose_generator(num_agents=NUM_AGENT)
    k = generation_data(test_data, generators, num_agents=NUM_AGENT)
    h = load_data(test_data, num_agents=NUM_AGENT)
    # battery characteristics
    s, c, d, e = create_battery_matrix(NUM_AGENT)

    # get the probability distribution for the solar and wind data
    prob_solar = get_probability_distribution(train_data, "Solar")
    prob_wind = get_probability_distribution(train_data, "Wind")

    # define models
    community_model = CommunityModel(1, s, c, d, e, NUM_AGENT, T)
    community_model_uncertain = CommunityModel(2, s, c, d, e, NUM_AGENT, T)

    # run the simulation
    mean_utility, mean_charging, savings, l_values, daily_utility, daily_charging, shapley_values, savings_2, exchange = (
        community_model.run_simulation(k, h, days=DAYS))
    (mean_utility_uncertain, mean_charging_uncertain, savings_uncertain, individual_mean_utility,
     individual_mean_charging, daily_utility_uncertain, daily_charging_uncertain, debt, shapley_values_uncertain, savings_2_uncertain, exchange_uncertain, charging_uncertain) = (
        community_model_uncertain.run_simulation_with_uncertainty(k, h, generators, prob_solar, prob_wind,
                                                                  l_values=l_values, days=DAYS))

    # calculate the distance between the actual and predicted values
    utility_distance = abs(daily_utility - daily_utility_uncertain)
    charging_distance = abs(daily_charging - daily_charging_uncertain)

    # compare the results
    compare_utility(mean_utility, mean_utility_uncertain, "Without Uncertainty", "With Uncertainty", NUM_AGENT,
                    file_name="Plots/utility_comparison.png")
    compare_charging(mean_charging, mean_charging_uncertain, "Without Uncertainty", "With Uncertainty", NUM_AGENT,
                     file_name="Plots/charging_comparison.png")
    compare_wasted_energy(savings, savings_uncertain, "Without Uncertainty", "With Uncertainty",
                          NUM_AGENT,
                          file_name="Plots/savings_comparison.png")
    compare_utility(mean_utility_uncertain, individual_mean_utility, "With Exchange", "Without Exchange",
                    NUM_AGENT, file_name="Plots/utility_comparison_individual.png")
    compare_charging(mean_charging_uncertain, individual_mean_charging, "With Exchange", "Without Exchange",
                     NUM_AGENT, file_name="Plots/charging_comparison_individual.png")
    compare_models_with_plot(daily_utility, daily_utility_uncertain, "Utility", "Utility with Uncertainty",
                             "Daily Utility", "Days", "Utility (kWh)", file_name="Plots/utility_difference.png")
    compare_models_with_plot(daily_charging, daily_charging_uncertain, "Charging", "Charging with Uncertainty",
                             "Daily Charging", "Days", "Charging (kWh)", file_name="Plots/charging_difference.png")
    compare_models_with_plot(utility_distance, charging_distance, "Utility Error", "Charging Error",
                             "Error Distance in Daily Utility and Charging", "Days", "Error Distance (kWh)",
                             file_name="Plots/error_distance_utility_charging.png")
    compare_models_with_plot(debt, None, "Total Daily Debt", None,
                             "Daily Debt", "Days", "Debt (kWh)",
                             file_name="Plots/debt_difference.png")
    show_saved_energy(savings_2_uncertain[0:50], "Without Uncertainty", 50, file_name="Plots/saved_energy.png")

    print(f"Total Exchange: {exchange}")
    print(f"Total Exchange with Uncertainty: {exchange_uncertain}")

    print(f"Total Charging with Uncertainty: {charging_uncertain}")
    print(f"Savings with Uncertainty: {np.sum(savings_2_uncertain)}")
