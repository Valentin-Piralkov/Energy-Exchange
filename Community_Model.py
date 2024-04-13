import pulp
import tabulate

from Data_Loader import load_data, compare_models_with_bar, display_agent_charging, display_agent_wasted_energy, \
    create_battery_matrix, generation_data, get_agent_utility, get_agent_charging, get_agent_wasted_energy
from Individual_Home import Homes

# Define constants
NUM_AGENTS = 10
t = 24
x = 0.001
battery_storage, battery_max_charge, battery_max_discharge, battery_efficiency = create_battery_matrix(NUM_AGENTS)


class Community_Model:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        # define the demand and generation
        self.k = generation_data(NUM_AGENTS)
        self.h = load_data(NUM_AGENTS)
        # Initialize the problem
        self.prob = pulp.LpProblem("Optimization_Problem", pulp.LpMinimize)
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
        self.l = pulp.LpVariable.dicts("l", [(j, i) for i in range(t) for j in range(NUM_AGENTS)], None, None,
                                       pulp.LpContinuous)
        # 1d array of saved energy values
        self.l_saved = pulp.LpVariable.dicts("l_saved", [i for i in range(t)], 0, None, pulp.LpContinuous)

    def get_individual_utility_and_charging(self):
        homes = Homes(1, self.k, self.h, battery_storage, battery_max_charge, battery_max_discharge, battery_efficiency)
        homes.optimise()
        return homes.get_utility_values(), homes.get_charging_values()

    def optimise(self):
        individual_utility_values, individual_charging_values = self.get_individual_utility_and_charging()

        # Define the objective function
        obj_vars = [self.c[j, i] for i in range(t) for j in range(NUM_AGENTS)]
        self.prob += pulp.lpSum(obj_vars)

        # Add constraints
        for j in range(0, NUM_AGENTS):
            for i in range(0, t):
                self.prob += self.p[j, i] == (self.g[j, i] - self.c[j, i] + self.d[j, i] + self.l[j, i]) / self.h[j, i]

        for j in range(0, NUM_AGENTS):
            self.prob += self.q[j, 0] == 5
            for i in range(1, t):
                self.prob += self.q[j, i] == self.q[j, i - 1] + battery_efficiency[j] * self.c[j, i - 1] - self.d[
                    j, i - 1]

        for j in range(0, NUM_AGENTS):
            for i in range(0, t):
                self.prob += self.p[j, i] == individual_utility_values[j, i]
                self.prob += self.q[j, i] >= 0
                self.prob += self.q[j, i] <= battery_storage[j]
                self.prob += self.c[j, i] >= 0
                self.prob += self.c[j, i] <= battery_max_charge[j]
                self.prob += self.d[j, i] >= 0
                self.prob += self.d[j, i] <= battery_max_discharge[j]
                self.prob += self.g[j, i] + self.w[j, i] == self.k[j, i]
                self.prob += 0 <= self.w[j, i] <= self.k[j, i]

        for i in range(0, t):
            self.prob += sum([self.c[j, i] for j in range(NUM_AGENTS)]) <= sum([individual_charging_values[j, i] for j
                                                                                in range(NUM_AGENTS)])
            self.prob += sum([self.l[j, i] for j in range(NUM_AGENTS)]) + self.l_saved[i] == 0
            self.prob += self.l_saved[i] >= 0

        self.prob += 0 < x

        # Solve the problem
        self.prob.solve()

        # display results
        print("Status:", pulp.LpStatus[self.prob.status])
        utility = []
        for j in range(0, NUM_AGENTS):
            u = 0
            for i in range(0, t):
                u += individual_utility_values[j, i]
            utility.append(u)
        self.compare_utility(utility, "Without Exchange")

    def compare_utility(self, other_model, other_label):
        compare_models_with_bar(NUM_AGENTS, get_agent_utility(NUM_AGENTS, t, self.p), other_model, "With Exchange",
                                other_label, "Agent Utilities", "Utility")

    def display_charging(self):
        display_agent_charging(NUM_AGENTS, t, self.c)

    def display_wasted_energy(self):
        display_agent_wasted_energy(NUM_AGENTS, t, self.w)

    def display_energy_flow(self):
        # display values of l as a table
        l_values = []
        for j in range(0, NUM_AGENTS):
            l_values.append([self.l[j, i].varValue for i in range(t)])
        print(tabulate.tabulate(l_values, headers=[f"Time {i}" for i in range(t)], tablefmt="fancy_grid"))

    def display_sum_energy_flow(self):
        # display the sum of l values at each time step t
        sum_l_values = [sum([self.l[j, i].varValue for j in range(NUM_AGENTS)]) for i in range(t)]
        print(tabulate.tabulate([sum_l_values], headers=[f"Time {i}" for i in range(t)], tablefmt="fancy_grid"))

    def __str__(self):
        return f"Agent {self.agent_id}:\n"


if __name__ == "__main__":
    community = Community_Model(1)
    community.optimise()
    community.display_charging()
    community.display_wasted_energy()
    community.display_energy_flow()
    community.display_sum_energy_flow()
