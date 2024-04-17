import numpy as np
import pulp

from Community_Utils import load_data, \
    create_battery_matrix, generation_data, get_agent_utility, get_agent_charging, \
    get_var_values, get_characteristic_functions, compare_utility, compare_charging
from Individual_Home import Homes
from Shapley import calculate_shapley_values, display_shapley_values


class CommunityModel:
    def __init__(self, model_id, num_agent=10, t=24):
        self.model_id = model_id
        self.num_agent = num_agent
        self.agent_ids = [i for i in range(1, num_agent + 1)]
        self.t = t
        self.q_values = np.zeros(num_agent)
        self.battery_storage, self.battery_max_charge, self.battery_max_discharge, self.battery_efficiency = (
            create_battery_matrix(num_agent))

    def get_individual_utility_and_charging(self, k, h):
        homes = Homes(1, k, h, self.battery_storage, self.battery_max_charge, self.battery_max_discharge,
                      self.battery_efficiency)
        homes.optimise()
        return homes.get_utility_values(), homes.get_charging_values(), homes.get_wasted_energy_values()

    def get_shapely_values(self, c):
        shap = calculate_shapley_values(1000, self.num_agent, c, self.t)
        return shap

    def optimise(self, k, h):
        individual_utility_values, individual_charging_values, individual_wasted_energy_values = (
            self.get_individual_utility_and_charging(k, h))
        # define the demand and generation
        # Initialize the problem
        prob = pulp.LpProblem("Optimization_Problem", pulp.LpMinimize)
        # Define the decision variables
        g = pulp.LpVariable.dicts("g", [(j, i) for i in range(self.t) for j in range(self.num_agent)], 0, 1.75,
                                  pulp.LpContinuous)
        w = pulp.LpVariable.dicts("w", [(j, i) for i in range(self.t) for j in range(self.num_agent)], 0, 1.75,
                                  pulp.LpContinuous)
        c = pulp.LpVariable.dicts("c", [(j, i) for i in range(self.t) for j in range(self.num_agent)], 0, 10,
                                  pulp.LpContinuous)
        d = pulp.LpVariable.dicts("d", [(j, i) for i in range(self.t) for j in range(self.num_agent)], 0, 10,
                                  pulp.LpContinuous)
        q = pulp.LpVariable.dicts("q", [(j, i) for i in range(self.t) for j in range(self.num_agent)], 0, 10,
                                  pulp.LpContinuous)
        p = pulp.LpVariable.dicts("p", [(j, i) for i in range(self.t) for j in range(self.num_agent)], 0, 1,
                                  pulp.LpContinuous)
        l = pulp.LpVariable.dicts("l", [(j, i) for i in range(self.t) for j in range(self.num_agent)], None, None,
                                  pulp.LpContinuous)
        # 1d array of saved energy values
        l_saved = pulp.LpVariable.dicts("l_saved", [i for i in range(self.t)], 0, None, pulp.LpContinuous)

        # Define the objective function
        obj_vars = [c[j, i] for i in range(self.t) for j in range(self.num_agent)]
        prob += pulp.lpSum(obj_vars)

        # Add constraints
        for j in range(0, self.num_agent):
            for i in range(0, self.t):
                prob += p[j, i] == (g[j, i] - c[j, i] + d[j, i] + l[j, i]) / h[j, i]

        for j in range(0, self.num_agent):
            prob += q[j, 0] == self.q_values[j]
            for i in range(1, self.t):
                prob += q[j, i] == q[j, i - 1] + self.battery_efficiency[j] * c[j, i - 1] - d[
                    j, i - 1]

        for i in range(0, self.t):
            prob += sum([c[j, i] for j in range(self.num_agent)]) <= sum([individual_charging_values[j, i] for j
                                                                          in range(self.num_agent)])
        for i in range(0, self.t):
            prob += l_saved[i] == -(sum([l[j, i] for j in range(self.num_agent)]))

        for j in range(0, self.num_agent):
            for i in range(0, self.t):
                prob += p[j, i] == individual_utility_values[j, i]
                prob += q[j, i] >= 0
                prob += q[j, i] <= self.battery_storage[j]
                prob += c[j, i] >= 0
                prob += c[j, i] <= self.battery_max_charge[j]
                prob += d[j, i] >= 0
                prob += d[j, i] <= self.battery_max_discharge[j]
                prob += g[j, i] + w[j, i] == k[j, i]
                prob += 0 <= w[j, i] <= k[j, i]

        # Solve the problem
        prob.solve()

        # display results
        print("Status:", pulp.LpStatus[prob.status])

        for j in range(0, self.num_agent):
            self.q_values[j] = q[j, self.t - 1].varValue

        return p, c, l_saved, individual_utility_values, individual_charging_values

    def run_simulation(self, days=5):
        k = generation_data(self.num_agent)
        h = load_data(self.num_agent)
        mean_utility = np.zeros(self.num_agent)
        mean_charging = np.zeros(self.num_agent)
        mean_utility_exchange = np.zeros(self.num_agent)
        mean_charging_exchange = np.zeros(self.num_agent)
        shapely_values = None
        for day in range(days):
            # take a 24hour slice of the data
            k_day = k[:, day * self.t:day * self.t + self.t]
            h_day = h[:, day * self.t:day * self.t + self.t]
            p, c, l_saved, individual_utility_values, individual_charging_values = self.optimise(k_day, h_day)
            mean_utility += get_var_values(individual_utility_values, self.num_agent, self.t)
            mean_charging += get_var_values(individual_charging_values, self.num_agent, self.t)
            mean_utility_exchange += get_agent_utility(self.num_agent, self.t, p)
            mean_charging_exchange += get_agent_charging(self.num_agent, self.t, c)
            optimal_charging, saved_energy = get_characteristic_functions(c, l_saved, self.num_agent, self.t)
            print(f"Charging: {optimal_charging}, Saved Energy: {saved_energy}")
            shapely_values = self.get_shapely_values(c)
            print(f"Shapley Values: {np.sum(shapely_values)}")
            print(f"Day {day + 1} completed")

        mean_utility /= days
        mean_charging /= days
        mean_utility_exchange /= days
        mean_charging_exchange /= days

        compare_utility(mean_utility_exchange, mean_utility, "Without Exchange", self.num_agent)
        compare_charging(mean_charging_exchange, mean_charging, "Without Exchange", self.num_agent)

        display_shapley_values(self.agent_ids, shapely_values, self.battery_storage)
