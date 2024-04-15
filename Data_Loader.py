import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# File paths
load_data_path = "Data/averaged_values.csv"


def load_data(num_agents=10):
    """
    Load the demand data from the csv file
    create a list of demand values for each agent
    add random noise to each agent's demand values
    :param num_agents: number of agents
    :return: matrix of demand values for each agent (h)
    """
    data = pd.read_csv(load_data_path)
    energy_values = data["Full demand"].values
    energy_matrix = np.zeros((num_agents, len(energy_values)))
    for i in range(num_agents):
        # add random noise to the demand values
        noise = np.random.normal(0, 0.2, len(energy_values))
        energy_matrix[i] = energy_values + noise
        # remove negative values
        energy_matrix[i][energy_matrix[i] < 0] = 0.0001
    return np.array(energy_matrix)


def generation_data(num_agents=10):
    """
    Load the generation data from the csv file
    create a list of generation values for each agent
    add random noise to each agent's generation values
    :param num_agents: number of agents
    :return: matrix of generation values for each agent (k)
    """
    data = pd.read_csv(load_data_path)
    generation_matrix = np.zeros((num_agents, len(data)))
    generators = choose_generator(num_agents)
    for i in range(num_agents):
        generator_name = generators[i]
        energy_values = data[generator_name].values
        # add random noise to the generation values
        noise = np.random.normal(0, 0.1, len(energy_values))
        generation_matrix[i] = energy_values + noise
        # remove negative values
        generation_matrix[i][generation_matrix[i] < 0] = 0
    return np.array(generation_matrix)


def get_agent_utility(num_agents, t, p):
    """
    return the daily utility (24h) for each agent
    the main purpose of this method is to prepare the data for bar chart plotting
    :param num_agents: number of agents
    :param t: number of time steps
    :param p: matrix of decision variables
    :return: list of utility values for each agent
    """
    utility = []
    for j in range(0, num_agents):
        u = 0
        for i in range(0, t):
            u += p[j, i].varValue
        utility.append(u)
    return utility


def get_agent_charging(num_agents, t, c):
    """
    return the daily charging (24h) for each agent
    the main purpose of this method is to prepare the data for bar chart plotting
    :param num_agents: number of agents
    :param t: number of time steps
    :param c: matrix of decision variables
    :return: list of charging values for each agent
    """
    charging = []
    for j in range(0, num_agents):
        u = 0
        for i in range(0, t):
            u += c[j, i].varValue
        charging.append(u)
    return charging


def get_agent_wasted_energy(num_agents, t, w):
    """
    return the daily wasted energy (24h) for each agent
    the main purpose of this method is to prepare the data for bar chart plotting
    :param num_agents: number of agents
    :param t: number of time steps
    :param w: matrix of decision variables
    :return: list of wasted energy values for each agent
    """
    wasted_energy = []
    for j in range(0, num_agents):
        u = 0
        for i in range(0, t):
            u += w[j, i].varValue
        wasted_energy.append(u)
    return wasted_energy


def compare_models_with_bar(num_agents, plot_1, plot_2, label_1, label_2, title, y_label):
    """
    display a bar chart comparing two models
    plots two bars for each agent
    allows for comparison between two results (e.g. with and without uncertainty)
    """
    labels = [i + 1 for i in range(0, num_agents)]
    # display the two model results next to each other
    x = np.arange(num_agents)
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, plot_1, width, label=label_1)
    ax.bar(x + width / 2, plot_2, width, label=label_2)
    plt.xlabel("Agent")
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def display_agent_charging(num_agents, t, c):
    """
    calculates the daily charging (24h) for each agent
    plots the charging for each agent in a bar chart
    """
    charging = []
    for j in range(0, num_agents):
        u = 0
        for i in range(0, t):
            u += c[j, i].varValue
        charging.append(u)
    # plot the charging
    labels = [i + 1 for i in range(0, num_agents)]
    plt.bar(labels, charging)
    plt.xlabel("Agent")
    plt.ylabel("Charging")
    plt.title("Agent Charging")
    plt.show()


def display_agent_wasted_energy(num_agents, t, w):
    """
    calculates the daily wasted energy (24h) for each agent
    plots the wasted energy for each agent in a bar chart
    """
    wasted_energy = []
    for j in range(0, num_agents):
        u = 0
        for i in range(0, t):
            u += w[j, i].varValue
        wasted_energy.append(u)
    # plot the wasted energy
    labels = [i + 1 for i in range(0, num_agents)]
    plt.bar(labels, wasted_energy)
    plt.xlabel("Agent")
    plt.ylabel("Wasted Energy")
    plt.title("Agent Wasted Energy")
    plt.show()


def create_battery_matrix(num_agents):
    """
    create a matrix of battery capacities for each agent
    battery capacities are chosen randomly based on 3 options
    those characteristics are passed separately: storage, max charge, max discharge, efficiency
    :param num_agents: number of agents
    :return: list of agent values for each battery characteristic (storage, charges, discharges, efficiency)
    """
    random_options = np.random.randint(1, 4, num_agents)

    battery_characteristics = []
    for i in range(num_agents):
        if random_options[i] == 1:
            battery_characteristics.append([10, 0.5, 0.5, 0.9])
        elif random_options[i] == 2:
            battery_characteristics.append([8, 0.4, 0.4, 0.9])
        else:
            battery_characteristics.append([6, 0.3, 0.3, 0.9])

    storage, charges, discharges, efficiency = [], [], [], []
    for i in range(num_agents):
        storage.append(battery_characteristics[i][0])
        charges.append(battery_characteristics[i][1])
        discharges.append(battery_characteristics[i][2])
        efficiency.append(battery_characteristics[i][3])

    return storage, charges, discharges, efficiency


def choose_generator(num_agents):
    """
    choose between a solar and a wind generator for each agent
    choices are made randomly
    :param num_agents: number of agents
    :return: list of generator names for each agent
    """
    generators = ["Solar CF", "Wind CF"]
    return np.random.choice(generators, num_agents)
