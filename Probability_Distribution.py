import pandas as pd
import numpy as np
from main import DISTANCE, MULTIPLIER

# test data
data = {'col': [1.8345, 2.3456, 1.8345, 3.4567, 2.3456, 1.8345, 3.4567, 1.763, 2.3456, 1.8345, 3.4567, 1.763, 2.345]}
df = pd.DataFrame(data)


def get_probability_distribution(col):
    """
    Get the probability distribution for a column in a dataframe
    :param col: the column to get the probability distribution for
    :return: a probability distribution series (index, probability)
    """
    # round values to 2 decimal places
    col = col.round(2)
    # get probabilities and normalise
    probabilities = col.value_counts(normalize=True)
    return probabilities


def weight_multiplier(value, dist=0.2, multiplier=2):
    """
    Weight multiplier function
    :param value: the probability index to be weighted
    :param dist: the distance threshold
    :param multiplier: the weight multiplier
    :return: 1 if the value is greater than the distance threshold, otherwise the multiplier
    """
    if value <= dist:
        return multiplier
    else:
        return 1


def choose_value_based_on_probability(probabilities, previous_value):
    """
    Choose a value based on the probability distribution
    Take into account the previous value and adjust the probabilities accordingly
    :param probabilities: the probability distribution
    :param previous_value: the previous value
    :return: a chosen value based on the adjusted probability distribution
    """
    previous_value = round(previous_value, 2)
    # get the distances between the probabilities and the previous value
    distances = abs(probabilities.index - previous_value)
    # adjust the probabilities based on the distances
    weights = [weight_multiplier(value, DISTANCE, MULTIPLIER) for value in distances]
    weighted_prob = probabilities * weights
    weighted_prob /= weighted_prob.sum()
    print(f"Weighted probabilities: {weighted_prob}")
    # choose a value based on the weighted probabilities
    chosen_value = np.random.choice(probabilities.index, p=weighted_prob)
    return chosen_value


if __name__ == '__main__':
    prob = get_probability_distribution(df['col'])
    print(f"Probability distribution: {prob}")
    c = choose_value_based_on_probability(prob, 1.8345)
    print(f"Chosen value: {c}")
