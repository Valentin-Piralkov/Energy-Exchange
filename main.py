import numpy as np
import pandas as pd
from tabulate import tabulate

# PARAMETERS
NUM_HOMES = 10  # Number of homes in the simulation
SIMULATION_DAYS = 30  # Duration of the simulation in days
TIME_STEPS_PER_DAY = 48  # 30 min time steps

# Initialize random seed for reproducibility
np.random.seed(42)

# Define homes with their properties
homes = pd.DataFrame({
    'id': range(1, NUM_HOMES + 1),
    'energy_generation': np.random.uniform(5, 15, NUM_HOMES),  # kWh generated per day
    'battery_capacity': np.random.uniform(10, 20, NUM_HOMES),  # kWh
    'daily_consumption': np.random.uniform(8, 12, NUM_HOMES),  # kWh consumed per day
    'battery_level': np.random.uniform(5, 10, NUM_HOMES),  # Initial battery level (kWh)
})


# Define a function to simulate a single day
def simulate_day(homes, day):
    # Reset daily exchange tracking
    homes['energy_received'] = 0.0
    homes['energy_shared'] = 0.0

    # Calculate daily generation and consumption
    homes['battery_level'] += homes['energy_generation'] - homes['daily_consumption']

    # Identify homes with excess energy and those in need
    excess_homes = homes[homes['battery_level'] > homes['battery_capacity']]
    needy_homes = homes[homes['battery_level'] < 0]

    # Simulate energy sharing
    for needy in needy_homes.itertuples():
        for excess in excess_homes.itertuples():
            if homes.at[needy.Index, 'battery_level'] < 0:
                shared_energy = min(excess.battery_level - excess.battery_capacity,
                                    -homes.at[needy.Index, 'battery_level'])
                homes.at[needy.Index, 'battery_level'] += shared_energy
                homes.at[excess.Index, 'battery_level'] -= shared_energy
                homes.at[needy.Index, 'energy_received'] += shared_energy
                homes.at[excess.Index, 'energy_shared'] += shared_energy

    # Ensure no battery overcharging
    homes['battery_level'] = np.minimum(homes['battery_level'], homes['battery_capacity'])
    homes['battery_level'] = np.maximum(homes['battery_level'], 0)  # Avoid negative battery levels


# Define a function to simulate a single day without energy exchange
def simulate_day_no_exchange(homes, day):
    # Calculate daily generation and consumption without exchange
    homes['battery_level'] += homes['energy_generation'] - homes['daily_consumption']

    # Ensure no battery overcharging or depletion below zero
    homes['battery_level'] = np.minimum(homes['battery_level'], homes['battery_capacity'])
    homes['battery_level'] = np.maximum(homes['battery_level'], 0)  # Avoid negative battery levels


# Create a copy of the homes DataFrame to simulate without exchange
homes_no_exchange = homes.copy()
homes_no_exchange['battery_level'] = homes_no_exchange['battery_capacity'] / 2  # Reset battery levels to 50% capacity

# Simulate each day
for day in range(SIMULATION_DAYS):
    simulate_day(homes, day)

# Display final state of homes
print(tabulate(homes, headers='keys', tablefmt='pretty'))

# Display total energy shared
print(homes['energy_shared'].sum())

# Display total energy received
print(homes['energy_received'].sum())

# Simulate each day without exchange
for day in range(SIMULATION_DAYS):
    simulate_day_no_exchange(homes_no_exchange, day)

# Plotting comparison
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Final Battery Levels With Exchange
axs[0].bar(homes['id'], homes['battery_level'], color='skyblue', label='With Exchange')
axs[0].set_title('Final Battery Levels With Exchange (kWh)')
axs[0].set_xlabel('Home ID')
axs[0].set_ylabel('Battery Level (kWh)')
axs[0].legend()

# Final Battery Levels Without Exchange
axs[1].bar(homes_no_exchange['id'], homes_no_exchange['battery_level'], color='tomato', label='Without Exchange')
axs[1].set_title('Final Battery Levels Without Exchange (kWh)')
axs[1].set_xlabel('Home ID')
axs[1].set_ylabel('Battery Level (kWh)')
axs[1].legend()

plt.tight_layout()
plt.show()




