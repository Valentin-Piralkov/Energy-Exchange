import random

import mesa
import seaborn as sns
import numpy as np
import pandas as pd

from Model.money_agent import MoneyAgent


def create_battery():
    option_1 = [10.0, 5.0, 5.0, 0.9]
    option_2 = [8.0, 4.0, 4.0, 0.9]
    option_3 = [6.0, 3.0, 3.0, 0.9]
    # choose between option 1, 2, 3 at random
    option = random.choice([option_1, option_2, option_3])
    return option


def create_generation(t=24):
    k = []
    for i in range(t):
        k.append(random.uniform(0, 1.75))
    return k


def create_load(t=24):
    h = []
    for i in range(t):
        h.append(random.uniform(0, 1.2))
    return h


class MoneyModel(mesa.Model):

    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            battery = create_battery()
            a = MoneyAgent(i, self, battery[0], battery[1], battery[2], battery[3], create_generation(), create_load())
            self.schedule.add(a)
            # Add the agent to a random grid cell

    def step(self):
        self.schedule.step()
