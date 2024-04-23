from Community_Model import CommunityModel
from Community_Utils import compare_utility, get_data, choose_generator, generation_data, load_data, compare_charging, \
    create_battery_matrix, get_var_values
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
    prob_solar = get_probability_distribution(train_data, "Solar CF")
    prob_wind = get_probability_distribution(train_data, "Wind CF")

    # define models
    community_model = CommunityModel(1, s, c, d, e, NUM_AGENT, T)
    community_model_uncertain = CommunityModel(2, s, c, d, e, NUM_AGENT, T)

    # run the simulation
    mean_utility, mean_charging, shapley_values = community_model.run_simulation(k, h, days=DAYS)
    mean_utility_uncertain, mean_charging_uncertain, shapley_values_uncertain, a, b = \
        (community_model_uncertain.run_simulation_with_uncertainty(k, h, generators, prob_solar, prob_wind, days=DAYS))

    # compare the results
    compare_utility(mean_utility, mean_utility_uncertain, "Without Uncertainty", "With Uncertainty", NUM_AGENT,
                    file_name="utility_comparison.png")
    compare_charging(mean_charging, mean_charging_uncertain, "Without Uncertainty", "With Uncertainty", NUM_AGENT,
                     file_name="charging_comparison.png")
    compare_utility(mean_utility_uncertain, a, "With Uncertainty", "Individual",
                    NUM_AGENT, file_name="utility_comparison_individual.png")
    compare_charging(mean_charging_uncertain, b, "With Uncertainty", "Individual",
                     NUM_AGENT, file_name="charging_comparison_individual.png")
