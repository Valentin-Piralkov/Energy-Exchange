import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Env import DAYS, DISTANCE, MULTIPLIER
from Community_Utils import get_data

# Set random seed
np.random.seed(42)


def get_probability_distribution(df, col_name):
    """
    Get the probability distribution for a column in a dataframe
    :param col_name:
    :param df:
    :return: a probability distribution series (index, probability)
    """
    # make an array of values for each time step
    time_values = {}
    for j in range(len(df)):
        t = df.iloc[j]["Time"]
        value = df.iloc[j][col_name]
        if t not in time_values:
            time_values[t] = []
        time_values[t].append(value)

    # round the values to 2 decimal places
    for t, values in time_values.items():
        time_values[t] = [round(value, 2) for value in values]

    # get the probability distribution for each time step
    probabilities = {}
    for t, values in time_values.items():
        value_counts = pd.Series(values).value_counts(normalize=True)
        probabilities[t] = value_counts

    return probabilities


def weight_multiplier(value, dist=0.2, multiplier=2, second_multiplier=1):
    """
    Weight multiplier function
    :param second_multiplier:
    :param value: the probability index to be weighted
    :param dist: the distance threshold
    :param multiplier: the weight multiplier
    :return: 1 if the value is greater than the distance threshold, otherwise the multiplier
    """
    if value <= dist:
        return multiplier
    else:
        return second_multiplier


def predict_next_day(probability_dict, previous_day, is_weighted=True):
    """
    Choose a value based on the probability distribution
    Take into account the previous value and adjust the probabilities accordingly
    :param is_weighted:
    :param previous_day:
    :param probability_dict:
    :return: a chosen value based on the adjusted probability distribution
    """
    next_day = []

    if not is_weighted:
        # choose a value based on the original probabilities
        for t, prob in probability_dict.items():
            next_value = np.random.choice(prob.index, p=prob.values)
            next_day.append(next_value)
        return np.array(next_day)

    # get the distances between the probabilities and the previous values
    distances = {}
    j = 0
    for t, prob in probability_dict.items():
        distances[t] = [abs(value - previous_day[j]) for value in prob.index]
        j += 1

    # choose a value based on the weighted probabilities
    for t, prob in probability_dict.items():
        weights = [weight_multiplier(dist, DISTANCE, MULTIPLIER) for dist in distances[t]]
        weighted_prob = prob * weights
        weighted_prob = weighted_prob / weighted_prob.sum()
        next_value = np.random.choice(weighted_prob.index, p=weighted_prob.values)
        next_day.append(next_value)

    return np.array(next_day)


def predict_next_day_wind(probability_dict, previous_value):
    """
    Choose a value based on the probability distribution
    Take into account the previous value and adjust the probabilities accordingly
    :param previous_value:
    :param probability_dict:
    :return: a chosen value based on the adjusted probability distribution
    """
    next_day = []

    # predict the next time step
    for t, prob in probability_dict.items():
        distances = [abs(value - previous_value) for value in prob.index]
        weights = [weight_multiplier(dist, DISTANCE - 0.1, MULTIPLIER, second_multiplier=0) for dist in distances]
        weighted_prob = prob * weights
        weighted_prob = weighted_prob / weighted_prob.sum()
        next_value = np.random.choice(weighted_prob.index, p=weighted_prob.values)
        next_day.append(next_value)
        previous_value = next_value

    return np.array(next_day)


if __name__ == '__main__':
    train_data, test_data = get_data()

    probabilities_solar = get_probability_distribution(test_data, "Solar")
    probabilities_wind = get_probability_distribution(test_data, "Wind")
    probabilities_demand = get_probability_distribution(test_data, "Full demand")
    previous_day_solar = train_data.tail(24)["Solar"].values
    previous_day_wind = train_data.tail(24)["Wind"].values
    previous_day_demand = train_data.tail(24)["Full demand"].values

    mse_list_solar = []
    mse_list_wind = []
    mse_list_demand = []
    hourly_mse_solar = np.zeros(24)
    hourly_wind = np.zeros(24)
    hourly_demand = np.zeros(24)
    next_day_solar = None
    next_day_wind = None
    next_day_demand = None
    actual_day_solar = None
    actual_day_wind = None
    actual_day_demand = None

    for i in range(DAYS):

        # predict the next day
        next_day_solar = predict_next_day(probabilities_solar, previous_day_solar)
        next_day_wind = predict_next_day_wind(probabilities_wind, previous_day_wind[-1])
        next_day_demand = predict_next_day(probabilities_demand, previous_day_demand)
        # get the actual day
        actual_day_solar = test_data.iloc[i * 24:(i + 1) * 24]["Solar"].values
        actual_day_wind = test_data.iloc[i * 24:(i + 1) * 24]["Wind"].values
        actual_day_demand = test_data.iloc[i * 24:(i + 1) * 24]["Full demand"].values
        # update the previous day
        previous_day_solar = actual_day_solar
        previous_day_wind = actual_day_wind
        previous_day_demand = actual_day_demand
        # calculate the mean squared error per hour
        hourly_mse_solar += np.square(np.subtract(actual_day_solar, next_day_solar))
        hourly_wind += np.square(np.subtract(actual_day_wind, next_day_wind))
        hourly_demand += np.square(np.subtract(actual_day_demand, next_day_demand))
        # calculate the mean squared error
        mse_solar = np.square(np.subtract(actual_day_solar, next_day_solar)).mean()
        mse_list_solar.append(mse_solar)
        mse_wind = np.square(np.subtract(actual_day_wind, next_day_wind)).mean()
        mse_list_wind.append(mse_wind)
        mse_demand = np.square(np.subtract(actual_day_demand, next_day_demand)).mean()
        mse_list_demand.append(mse_demand)

    # scale the mean squared error per hour
    hourly_mse_solar = hourly_mse_solar / DAYS
    hourly_wind = hourly_wind / DAYS
    hourly_demand = hourly_demand / DAYS

    print("Mean Squared Error for Solar Generation: ", np.mean(mse_list_solar))
    print("Mean Squared Error for Wind Generation: ", np.mean(mse_list_wind))
    print("Mean Squared Error for Consumption: ", np.mean(mse_list_demand))

    # plot the mean squared error per hour
    plt.plot(hourly_mse_solar, label="Solar")
    plt.plot(hourly_wind, label="Wind")
    plt.plot(hourly_demand, label="Full demand")
    plt.xlabel("Time (h)")
    plt.ylabel("Mean Squared Error")
    plt.title("Hourly Mean Squared Error")
    plt.legend(loc="upper left")
    plt.savefig("Plots/Hourly_Mean_Squared_Error.png")
    plt.show()

    # plot the mean squared error
    plt.plot(mse_list_solar, label="Solar")
    plt.plot(mse_list_wind, label="Wind")
    plt.plot(mse_list_demand, label="Consumption")
    plt.xlabel("Day")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Squared Error for Generation and Consumption Predictions")
    plt.legend(loc="upper left")
    plt.savefig("Plots/Mean_Squared_Error.png")
    plt.show()

    # plot the actual vs predicted values with bar chart
    x = np.arange(24)
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, actual_day_solar, width, label="Actual")
    ax.bar(x + width / 2, next_day_solar, width, label="Predicted")
    plt.xlabel("Time (h)")
    plt.ylabel("Solar Generation (kWh)")
    plt.title("Actual vs Predicted Solar Generation")
    plt.legend(loc="upper left")
    plt.savefig("Plots/Actual_vs_Predicted_Solar_CF.png")
    plt.show()

    # noinspection PyRedeclaration
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, actual_day_wind, width, label="Actual")
    ax.bar(x + width / 2, next_day_wind, width, label="Predicted")
    plt.xlabel("Time (h)")
    plt.ylabel("Wind Generation (kWh)")
    plt.title("Actual vs Predicted Wind Generation")
    plt.legend(loc="upper left")
    plt.savefig("Plots/Actual_vs_Predicted_Wind_CF.png")
    plt.show()

    # noinspection PyRedeclaration
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, actual_day_demand, width, label="Actual")
    ax.bar(x + width / 2, next_day_demand, width, label="Predicted")
    plt.xlabel("Time (h)")
    plt.ylabel("Consumption (kWh)")
    plt.title("Actual vs Predicted Consumption")
    plt.legend(loc="upper left")
    plt.savefig("Plots/Actual_vs_Predicted_Full_Demand.png")
    plt.show()

    # plot the actual vs predicted values with line chart
    plt.plot(actual_day_solar, label="Actual")
    plt.plot(next_day_solar, label="Predicted")
    plt.xlabel("Time (h)")
    plt.ylabel("Solar Generation (kWh)")
    plt.title("Actual vs Predicted Solar Generation")
    plt.legend(loc="upper left")
    plt.savefig("Plots/Actual_vs_Predicted_Solar_CF_Line.png")
    plt.show()

    plt.plot(actual_day_wind, label="Actual")
    plt.plot(next_day_wind, label="Predicted")
    plt.ylim(bottom=0)
    plt.xlabel("Time (h)")
    plt.ylabel("Wind Generation (kWh)")
    plt.title("Actual vs Predicted Wind Generation")
    plt.legend(loc="lower left")
    plt.savefig("Plots/Actual_vs_Predicted_Wind_CF_Line.png")
    plt.show()

    plt.plot(actual_day_demand, label="Actual")
    plt.plot(next_day_demand, label="Predicted")
    plt.ylim(bottom=0)
    plt.xlabel("Time (h)")
    plt.ylabel("Consumption (kWh)")
    plt.title("Actual vs Predicted Consumption")
    plt.legend(loc="lower left")
    plt.savefig("Plots/Actual_vs_Predicted_Full_Demand_Line.png")
    plt.show()
