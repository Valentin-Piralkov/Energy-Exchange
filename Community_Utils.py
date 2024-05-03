import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Env import LOAD_DATA_PATH

np.random.seed(42)


def get_data(split=0.8):
    """
    Load the demand and generation data from the csv file
    :return: the demand and generation data
    """
    data = pd.read_csv(LOAD_DATA_PATH)
    # split the data into test and train
    train_data = data.iloc[:int(split * len(data))]
    test_data = data.iloc[int(split * len(data)):]

    return train_data, test_data


def load_data(data, num_agents=10):
    """
    Load the demand data from the csv file
    create a list of demand values for each agent
    add random noise to each agent's demand values
    :param data:
    :param num_agents: number of agents
    :return: matrix of demand values for each agent (h)
    """
    energy_values = data["Full demand"].values
    energy_matrix = np.zeros((num_agents, len(energy_values)))
    for i in range(num_agents):
        # add random noise to the demand values
        noise = np.random.normal(0, 0.2, len(energy_values))
        energy_matrix[i] = energy_values + noise
        # remove negative values
        energy_matrix[i][energy_matrix[i] < 0] = 0.0001
    return np.array(energy_matrix)


def generation_data(data, generators, num_agents=10):
    """
    Load the generation data from the csv file
    create a list of generation values for each agent
    add random noise to each agent's generation values
    :param generators:
    :param data:
    :param num_agents: number of agents
    :return: matrix of generation values for each agent (k)
    """
    generation_matrix = np.zeros((num_agents, len(data)))
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


def compare_models_with_bar(num_agents, plot_1, plot_2, label_1, label_2, title, y_label, file_name=None):
    """
    display a bar chart comparing two models
    plots two bars for each agent
    allows for comparison between two results (e.g. with and without uncertainty)
    """
    x = np.arange(num_agents)
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, plot_1, width, label=label_1)
    ax.bar(x + width / 2, plot_2, width, label=label_2)
    plt.xlabel("Agent")
    plt.ylabel(y_label)
    plt.title(title)
    ax.legend(loc='upper right')
    if file_name:
        plt.savefig(file_name)
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
    generators = ["Solar", "Wind"]
    # randomly choose between solar and wind, solar has greater probability
    generator_choices = np.random.choice(generators, num_agents, p=[0.5, 0.5])
    return generator_choices


def get_var_values(variables, num_agents, t):
    """
    return the sum of the decision variables for each agent
    :param variables: the decision variables
    :param num_agents: the number of agents
    :param t: the number of time steps
    :return: a list of the sum of the decision variables for each agent
    """
    variable_list = []
    for j in range(0, num_agents):
        u = 0
        for i in range(0, t):
            u += variables[j, i]
        variable_list.append(u)
    return variable_list


def compare_utility(p, other_model, label, other_label, num_agents, file_name=None):
    """
    compare the utility values of two models
    :param file_name: name of the file to save the chart to
    :param label: the label for the first model
    :param p: the utility values of the first model
    :param other_model: the utility values of the second model
    :param other_label: the label for the second model
    :param num_agents: the number of agents in the two models
    :return: display a bar chart comparing the utility values of the two models
    """
    compare_models_with_bar(num_agents, p, other_model, label,
                            other_label, "Agent Mean Utilities", "Utility", file_name)


def compare_charging(c, other_model, label, other_label, num_agents, file_name=None):
    """
    compare the charging values of two models
    :param file_name: name of the file to save the chart to
    :param label: the label for the first model
    :param c: the charging values of the first model
    :param other_model: the charging values of the second model
    :param other_label: the label for the second model
    :param num_agents: the number of agents in the two models
    :return: display a bar chart comparing the charging values of the two models
    """
    compare_models_with_bar(num_agents, c, other_model, label,
                            other_label, "Agent Mean Charging", "Charging (kWh)", file_name)


def compare_wasted_energy(w, other_model, label, other_label, num_agents, file_name=None):
    """
    compare the wasted energy values of two models
    :param file_name: name of the file to save the chart to
    :param label: the label for the first model
    :param w: the wasted energy values of the first model
    :param other_model: the wasted energy values of the second model
    :param other_label: the label for the second model
    :param num_agents: the number of agents in the two models
    :return: display a bar chart comparing the wasted energy values of the two models
    """
    compare_models_with_bar(num_agents, w, other_model, label,
                            other_label, "Agent Mean Savings", "Energy Savings (kWh)", file_name)


def get_characteristic_functions(c, l_saved, u, num_agents, t):
    """
    return the characteristic functions of the agents
    the first function returns the total charging values of all agents
    the second function returns the total saved energy values of all agents
    :param u: utility values of the agents
    :param c: the charging values of the agents
    :param l_saved: the saved energy values of the agents
    :param num_agents: the number of agents
    :param t: the number of time steps
    :return: the total charging values of all agents, the total saved energy values of all agents
    """
    v_c = sum([c[j, i].varValue for i in range(t) for j in range(num_agents)])
    v_e = sum([l_saved[i].varValue for i in range(t)]) / 10  # normalize the saved energy
    v_u = sum([u[j, i].varValue for i in range(t) for j in range(num_agents)])
    return v_c, v_e, v_u


def compare_models_with_plot(plot_1, plot_2, label_1, label_2, title, x_label, y_label, file_name=None):
    """
    display a line chart comparing two models
    plots two lines for each agent
    allows for comparison between two results (e.g. with and without uncertainty)
    """
    # add custom x labels

    plt.plot(plot_1, label=label_1)
    plt.xticks(range(len(plot_1)), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    if plot_2 is not None:
        plt.plot(plot_2, label=label_2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if plot_2 is not None:
        plt.legend()
    if file_name:
        plt.savefig(file_name)
    plt.show()


def show_saved_energy(l_saved_1, label_1, days, file_name=None):
    """
    calculates the daily saved energy (24h) for each agent
    plots the saved energy for each agent in a bar chart
    """
    labels = [(i + 1) for i in range(0, days)]
    plt.bar(labels, l_saved_1)
    plt.xlabel("Days")
    plt.ylabel("Saved Energy (kWh)")
    plt.title("Daily Saved Energy")
    if file_name:
        plt.savefig(file_name)
    plt.show()


def display_bar_chart(data, title, x_label, y_label, file_name=None):
    """
    display a bar chart
    :param data: the data to display
    :param title: the title of the chart
    :param x_label: the x-axis label
    :param y_label: the y-axis label
    :param file_name: the name of the file to save the chart to
    """
    labels = [i + 1 for i in range(0, len(data))]
    plt.bar(labels, data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if file_name:
        plt.savefig(file_name)
    plt.show()
