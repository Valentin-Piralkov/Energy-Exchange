from itertools import permutations
import random
import numpy as np
from matplotlib import pyplot as plt


def calculate_permutations(agents, limit=121):
    """
    Calculates all possible permutations for a set of agents
    :param agents: list of agents
    :param limit: when the number of agents is greater than 10, limit the number of permutations to calculate
    :return: list of permutations
    """
    perms = []
    if len(agents) > 10:
        samples = len(agents) * limit
        print(f"Calculating {samples} permutations")

        for permutation in permutations(agents):
            perms.append(permutation)
            if len(perms) >= samples:
                break
    else:
        for permutation in permutations(agents):
            perms.append(permutation)
    return perms


def calculate_predecessors(p, agent):
    """
    Calculates the predecessors of an agent in a permutation
    :param p: the permutation
    :param agent: current agent
    :return: predecessors of the agent in the permutation
    """
    pred = []
    for i in range(len(p)):
        pred.append(p[i])
        if p[i] == agent:
            break
    return pred


def calculate_contribution(pred, agent, c, t):
    """
    Calculates the contribution of an agent in a permutation
    :param pred: the predecessors of the agent
    :param agent: the current agent
    :param c: charging or saved energy values
    :param t: time steps
    :return: contribution of the agent
    """
    pred_i = pred + [agent]
    if len(pred) == 0:
        return 0

    v_c_i = sum([c[j, i].varValue for i in range(t) for j in pred_i])
    v_c_pred = sum([c[j, i].varValue for i in range(t) for j in pred])
    return v_c_i - v_c_pred


def calculate_shapley_values(m, n, c, t):
    """
    Calculates the Shapley values for a set of agents
    :param m: loop count
    :param n: number of agents
    :param c: characteristics function
    :param t: time steps
    :return: Shapley values
    """
    agents = [i for i in range(n)]
    perms = calculate_permutations(agents)
    count = 0
    shap = np.zeros(n)
    while count < m:
        # choose a random permutation
        perm = random.choice(perms)
        for i in range(n):
            pred = calculate_predecessors(perm, i + 1)
            contribution = calculate_contribution(pred, i, c, t)
            shap[i] += contribution
        count += 1

    shap = shap / m
    return shap


def display_shapley_values(agents, shap, storage):
    # color each bar based on the storage capacity
    colors = []
    for i in range(len(shap)):
        if storage[i] == 10:
            colors.append("red")
        elif storage[i] == 8:
            colors.append("blue")
        else:
            colors.append("green")
    plt.bar(agents, shap, color=colors)
    plt.xlabel("Agent")
    plt.ylabel("Shapley Value")
    plt.title("Shapley Values")
    plt.show()
