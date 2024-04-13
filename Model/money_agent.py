import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import pulp


class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, s, c_max, d_max, e, k, h, x=0.001, t=24):
        super().__init__(unique_id, model)
        self.prob = None
        self.e = e
        self.s = s
        self.c_max = c_max
        self.d_max = d_max
        self.x = x
        self.t = t
        self.k = k
        self.h = h

    def optimise(self):
        # Initialize the problem
        self.prob = pulp.LpProblem("Optimization_Problem", pulp.LpMaximize)

        # Define the decision variables
        g = pulp.LpVariable.dicts("g", range(0, self.t), 0, 1.8, pulp.LpContinuous)
        w = pulp.LpVariable.dicts("w", range(0, self.t), 0, 1.8, pulp.LpContinuous)
        c = pulp.LpVariable.dicts("c", range(0, self.t), 0, self.c_max, pulp.LpContinuous)
        d = pulp.LpVariable.dicts("d", range(0, self.t), 0, self.d_max, pulp.LpContinuous)
        q = pulp.LpVariable.dicts("q", range(self.t), 0, self.s, pulp.LpContinuous)
        p = pulp.LpVariable.dicts("p", range(0, self.t), 0, 1, pulp.LpContinuous)
        l = pulp.LpVariable.dicts("l", range(0, self.t), None, None, pulp.LpContinuous)

        # Define the objective function
        obj_vars = [p[i] - self.x * c[i] for i in range(0, self.t)]
        self.prob += pulp.lpSum(obj_vars)

        # Add constraints
        for i in range(0, self.t):
            self.prob += p[i] == (g[i] - c[i] + d[i] + l[i]) / self.h[i]

        for i in range(1, self.t):
            self.prob += q[i] == q[i - 1] + self.e * c[i - 1] - d[i - 1]
        self.prob += q[0] == q[self.t - 1] + self.e * c[self.t - 1] - d[self.t - 1]

        for i in range(0, self.t):
            self.prob += q[i] >= 0
            self.prob += q[i] <= self.s
            self.prob += c[i] >= 0
            self.prob += c[i] <= self.c_max
            self.prob += d[i] >= 0
            self.prob += d[i] <= self.d_max
            self.prob += self.k[i] == g[i] + w[i]
            self.prob += 0 <= w[i] <= self.k[i]

        self.prob += 0 <= self.e <= 1
        self.prob += 0 < self.s
        self.prob += 0 < self.c_max
        self.prob += 0 < self.d_max
        self.prob += 0 < self.x

    def solve(self):
        self.prob.solve()

    def step(self):
        self.optimise()
        if self.prob is not None:
            self.solve()
        else:
            print("Problem not initialized")
