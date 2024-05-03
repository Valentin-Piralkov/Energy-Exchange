import numpy as np
import pulp

from Community_Utils import get_agent_utility, get_agent_charging, get_characteristic_functions, get_var_values
from Individual_Home import Homes
from Probability_Distribution import predict_next_day, predict_next_day_wind
from Shapley import calculate_shapley_values


class CommunityModel:
    def __init__(self, model_id, battery_storage, battery_max_charge, battery_max_discharge, battery_efficiency,
                 num_agent, t):
        self.model_id = model_id
        self.num_agent = num_agent
        self.agent_ids = [i for i in range(1, num_agent + 1)]
        self.t = t
        self.q_values = np.zeros(num_agent)
        self.debt = np.zeros((num_agent, t))
        self.battery_storage = battery_storage
        self.battery_max_charge = battery_max_charge
        self.battery_max_discharge = battery_max_discharge
        self.battery_efficiency = battery_efficiency

    def get_individual_utility_and_charging(self, k, h):
        homes = Homes(1, k, h, self.battery_storage, self.battery_max_charge, self.battery_max_discharge,
                      self.battery_efficiency, self.num_agent, self.t)
        homes.optimise()
        return homes.get_utility_values(), homes.get_charging_values(), homes.get_wasted_energy_values()

    def get_shapely_values(self, c):
        shap = calculate_shapley_values(1000, self.num_agent, c, self.t)
        return shap

    def optimise(self, k, h, l_values=None):
        individual_utility_values, individual_charging_values, individual_wasted_energy_values = (
            self.get_individual_utility_and_charging(k, h))
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
        t_l = pulp.LpVariable.dicts("t_l", [(j, i) for i in range(self.t) for j in range(self.num_agent)], None, None,
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
            prob += sum([l[j, i] for j in range(self.num_agent)]) + l_saved[i] == 0

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
                prob += l[j, i] == self.debt[j, i] + t_l[j, i]
                prob += 0 <= w[j, i] <= k[j, i]

        # Solve the problem
        prob.solve()

        # display results
        print("Status:", pulp.LpStatus[prob.status])

        for j in range(0, self.num_agent):
            self.q_values[j] = q[j, self.t - 1].varValue

        if l_values is not None:
            for j in range(0, self.num_agent):
                for i in range(0, self.t):
                    self.debt[j, i] = -(l[j, i].varValue - l_values[j, i].varValue)

        return p, c, l_saved, l, individual_utility_values, individual_charging_values

    def run_simulation(self, k, h, days=30):

        if self.num_agent <= 0:
            return (None, None, None, None,
                    None, None, None, None, None, None)

        # define arrays to store the mean values
        mean_utility = np.zeros(self.num_agent)
        mean_charging = np.zeros(self.num_agent)
        daily_utility = []
        daily_charging = []
        l_values = []
        savings = np.zeros(self.num_agent)
        mean_charging_individual = np.zeros(self.num_agent)
        savings_2 = []
        exchange = 0

        shapely_values = None

        # iterate through the days
        for day in range(1, days + 1):
            # take a 24hour slice of the data
            k_day = k[:, day * self.t:day * self.t + self.t]
            h_day = h[:, day * self.t:day * self.t + self.t]

            # optimize the model for the actual and predicted data
            p, c, l_saved, l, _, individual_c = self.optimise(k_day, h_day)
            l_values.append(l)
            for j in range(self.num_agent):
                for i in range(self.t):
                    value = l[j, i].varValue
                    exchange += value
                    if value < 0:
                        savings[j] += 0
                    else:
                        savings[j] += l[j, i].varValue

            # get the utility and charging values
            mean_utility += get_agent_utility(self.num_agent, self.t, p)
            mean_charging += get_agent_charging(self.num_agent, self.t, c)
            mean_charging_individual += get_var_values(individual_c, self.num_agent, self.t)

            # get the characteristic functions
            optimal_charging, saved_energy, total_utility = get_characteristic_functions(c, l_saved, p,
                                                                                         self.num_agent, self.t)
            daily_utility.append(total_utility)
            daily_charging.append(optimal_charging)
            savings_2.append(saved_energy)

            # display the results
            print(f"Charging: {optimal_charging}, Saved Energy: {saved_energy}, Total Utility: {total_utility}")
            shapely_values = self.get_shapely_values(c)
            print(f"Shapley Values: {np.sum(shapely_values)}")
            print(f"Day {day} completed")
            if day != days:
                print()
                print("--------------------------------------------------")
                print()

        # calculate the mean values
        mean_utility /= days
        mean_charging /= days
        mean_charging_individual /= days

        daily_utility = np.array(daily_utility) / self.num_agent
        daily_charging = np.array(daily_charging) / self.num_agent
        savings /= days
        exchange = abs(exchange)

        return (mean_utility, mean_charging, savings, np.array(l_values),
                daily_utility, daily_charging, shapely_values, savings_2, exchange, mean_charging_individual)

    def run_simulation_with_uncertainty(self, k, h, generators, prob_solar, prob_wind, l_values, days=30):
        if self.num_agent <= 0:
            if self.num_agent <= 0:
                return (None, None, None, None,
                        None, None, None, None, None, None, None, None)

        # define arrays to store the mean values
        mean_utility = np.zeros(self.num_agent)
        mean_charging = np.zeros(self.num_agent)
        mean_utility_individual = np.zeros(self.num_agent)
        mean_charging_individual = np.zeros(self.num_agent)
        debt_values = np.zeros(days)
        shapley_values = None
        daily_utility = []
        daily_charging = []
        savings = np.zeros(self.num_agent)
        savings_2 = []
        exchange = 0
        charging = np.zeros(2)

        # iterate through the days
        for day in range(1, days + 1):
            # take a 24hour slice of the data
            h_day = h[:, day * self.t:day * self.t + self.t]
            # k_day_actual = k[:, day * self.t:day * self.t + self.t]

            # predict the next day
            k_day_predicted = []
            for i in range(self.num_agent):
                previous_day = k[i, (day - 1) * self.t:day * self.t]
                if generators[i] == "Solar":
                    k_day_predicted.append(predict_next_day(prob_solar, previous_day))
                elif generators[i] == "Wind":
                    k_day_predicted.append(predict_next_day_wind(prob_wind, previous_day[-1]))
                else:
                    raise ValueError("Invalid generator")
            k_day_predicted = np.array(k_day_predicted)

            # optimize the model for the actual and predicted data
            if l_values is None or day == 1:
                p, c, l_saved, l, individual_p, individual_c = self.optimise(k_day_predicted, h_day)
            else:
                p, c, l_saved, l, individual_p, individual_c = self.optimise(k_day_predicted, h_day,
                                                                             l_values=l_values[day - 1])

            debt_values[day - 1] = np.sum(self.debt)

            for j in range(self.num_agent):
                for i in range(self.t):
                    value = l[j, i].varValue
                    exchange += value
                    if value < 0:
                        savings[j] += 0
                    else:
                        savings[j] += l[j, i].varValue

            # get the utility and charging values
            mean_utility += get_agent_utility(self.num_agent, self.t, p)
            mean_charging += get_agent_charging(self.num_agent, self.t, c)
            mean_utility_individual += get_var_values(individual_p, self.num_agent, self.t)
            mean_charging_individual += get_var_values(individual_c, self.num_agent, self.t)

            # get the characteristic functions
            optimal_charging, saved_energy, total_utility = (
                get_characteristic_functions(c, l_saved, p, self.num_agent, self.t))

            daily_utility.append(total_utility)
            daily_charging.append(optimal_charging)
            savings_2.append(saved_energy)

            # display the results
            print(f"Charging: {optimal_charging}, Saved Energy: {saved_energy}, Total Utility: {total_utility}")
            shapley_values = self.get_shapely_values(c)
            print(f"Shapley Values: {np.sum(shapley_values)}")
            print(f"Day {day} completed")
            if day != days:
                print()
                print("--------------------------------------------------")
                print()

        # calculate the mean values
        mean_utility /= days
        mean_charging /= days
        mean_utility_individual /= days
        mean_charging_individual /= days
        charging[0] = np.sum(mean_charging)
        charging[1] = np.sum(mean_charging_individual)
        daily_utility = np.array(daily_utility) / self.num_agent
        daily_charging = np.array(daily_charging) / self.num_agent
        savings /= days
        exchange = abs(exchange)

        print(f"Debt: {debt_values}")


        return (mean_utility, mean_charging, savings, mean_utility_individual, mean_charging_individual,
                daily_utility, daily_charging, debt_values, shapley_values, savings_2, exchange, charging)
