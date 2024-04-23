import torch
import torch.nn as nn
import torch.distributions as distributions

# 1. Sample Dataset (replace with your actual data)
import numpy as np
x_train = torch.from_numpy(np.random.rand(1000, 5)).float()
y_train = torch.from_numpy(np.random.randn(1000) + 2*x_train[:, 0]).float()

# 2. Model Architecture (adjust for your problem)
class ProbabilisticModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define 2 hidden layers
        self.hidden = nn.Linear(5, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = nn.functional.relu(self.hidden(x))
        mean = self.out(x)[:, 0]
        std = nn.functional.softplus(self.out(x)[:, 1])
        return distributions.Normal(mean, std)

model = ProbabilisticModel()

# 3. Probabilistic Loss
def probabilistic_loss(y_true, y_pred):
    return -y_pred.log_prob(y_true).mean()

optimizer = torch.optim.Adam(model.parameters())

# 4. Training Loop
for epoch in range(100):
    pred_dist = model(x_train)
    loss = probabilistic_loss(y_train, pred_dist)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Usage (Prediction with Uncertainty)
new_x = torch.from_numpy(np.random.rand(1, 5)).float()
pred_dist = model(new_x)
pred_mean = pred_dist.mean.detach().numpy()
pred_std = pred_dist.stddev.detach().numpy()

print(f"Predicted mean: {pred_mean}")
print(f"Predicted standard deviation: {pred_std}")
