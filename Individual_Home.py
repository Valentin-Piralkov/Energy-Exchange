import numpy as np
import pulp

# Define constants
NUM_AGENTS = 10
t = 24
x = 0.001


class Homes:
    def __init__(self, agent_id, k, h, battery_storage, battery_max_charge, battery_max_discharge, battery_efficiency):
        self.agent_id = agent_id
        # define battery characteristics
        self.battery_storage = battery_storage
        self.battery_max_charge = battery_max_charge
        self.battery_max_discharge = battery_max_discharge
        self.battery_efficiency = battery_efficiency
        # define the demand and generation
        self.k = k
        self.h = h
        # Initialize the problem
        self.prob = pulp.LpProblem("Optimization_Problem", pulp.LpMaximize)
        # Define the decision variables
        self.g = pulp.LpVariable.dicts("g", [(j, i) for i in range(t) for j in range(NUM_AGENTS)], 0, 1.75,
                                       pulp.LpContinuous)
        self.w = pulp.LpVariable.dicts("w", [(j, i) for i in range(t) for j in range(NUM_AGENTS)], 0, 1.75,
                                       pulp.LpContinuous)
        self.c = pulp.LpVariable.dicts("c", [(j, i) for i in range(t) for j in range(NUM_AGENTS)], 0, 10,
                                       pulp.LpContinuous)
        self.d = pulp.LpVariable.dicts("d", [(j, i) for i in range(t) for j in range(NUM_AGENTS)], 0, 10,
                                       pulp.LpContinuous)
        self.q = pulp.LpVariable.dicts("q", [(j, i) for i in range(t) for j in range(NUM_AGENTS)], 0, 10,
                                       pulp.LpContinuous)
        self.p = pulp.LpVariable.dicts("p", [(j, i) for i in range(t) for j in range(NUM_AGENTS)], 0, 1,
                                       pulp.LpContinuous)

    def optimise(self):
        # Define the objective function
        obj_vars = [self.p[j, i] - x * self.c[j, i] for i in range(t) for j in range(NUM_AGENTS)]
        self.prob += pulp.lpSum(obj_vars)

        # Add constraints
        for j in range(0, NUM_AGENTS):
            for i in range(0, t):
                self.prob += self.p[j, i] == (self.g[j, i] - self.c[j, i] + self.d[j, i]) / self.h[j, i]

        for j in range(0, NUM_AGENTS):
            self.prob += self.q[j, 0] == 2
            for i in range(1, t):
                self.prob += self.q[j, i] == self.q[j, i - 1] + self.battery_efficiency[j] * self.c[j, i - 1] - self.d[
                    j, i - 1]

        for j in range(0, NUM_AGENTS):
            for i in range(0, t):
                self.prob += self.q[j, i] >= 0
                self.prob += self.q[j, i] <= self.battery_storage[j]
                self.prob += self.c[j, i] >= 0
                self.prob += self.c[j, i] <= self.battery_max_charge[j]
                self.prob += self.d[j, i] >= 0
                self.prob += self.d[j, i] <= self.battery_max_discharge[j]
                self.prob += self.g[j, i] + self.w[j, i] == self.k[j, i]
                self.prob += 0 <= self.w[j, i] <= self.k[j, i]

        self.prob += 0 < x

        # Solve the problem
        self.prob.solve()

    def get_utility_values(self):
        # return the p values as a np array
        return np.array([[self.p[j, i].varValue for i in range(t)] for j in range(NUM_AGENTS)])

    def get_charging_values(self):
        # return the c values as a np array
        return np.array([[self.c[j, i].varValue for i in range(t)] for j in range(NUM_AGENTS)])

    def get_wasted_energy_values(self):
        # return the w values as a np array
        return np.array([[self.w[j, i].varValue for i in range(t)] for j in range(NUM_AGENTS)])

