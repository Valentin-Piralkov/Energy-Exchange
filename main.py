import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from Model.money_model import MoneyModel

# Define constants
NUM_AGENTS = 10
NUM_STEPS = 5

model = MoneyModel(NUM_AGENTS)
for i in range(NUM_STEPS):
    model.step()

agent_1 = model.schedule.agents[0]

print(agent_1.e)
g_values = [agent_1.g[i].varValue for i in range(0, 24)]
print(g_values)
