from Community_Model import CommunityModel

# Define Constants
MODEL_ID = 1
NUM_AGENT = 10
T = 24
DAYS = 120
DISTANCE = 0.2
MULTIPLIER = 20

if __name__ == '__main__':
    community_model = CommunityModel(MODEL_ID, NUM_AGENT, T)
    community_model.run_simulation(DAYS)
