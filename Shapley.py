import itertools
import random
import numpy as np
from matplotlib import pyplot as plt


def calculate_permutations(agents):
    p = list(itertools.permutations(agents))
    return p


def calculate_predecessors(p, agent):
    pred = []
    for i in range(len(p)):
        pred.append(p[i])
        if p[i] == agent:
            break
    return pred


def calculate_contribution(pred, agent, c, t):
    pred_i = pred + [agent]
    if len(pred) == 0:
        return 0

    v_c_i = sum([c[j, i].varValue for i in range(t) for j in pred_i])
    v_c_pred = sum([c[j, i].varValue for i in range(t) for j in pred])
    return v_c_i - v_c_pred


def calculate_shapley_values(m, n, c, t):
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
