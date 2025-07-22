import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data.datasets import D4RLExperienceReplay

dataset = D4RLExperienceReplay(dataset_id="hopper-medium-v0", batch_size=256)

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

batch = next(iter(dataset))
obs_dim = batch["observation"].shape[-1]
act_dim = batch["action"].shape[-1]
q_net = QNet(obs_dim, act_dim)
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
gamma = 0.99

batch = next(iter(dataset))
obs = batch["observation"]
act = batch["action"]
rew = batch["reward"].unsqueeze(-1)
next_obs = batch["next", "observation"]
done = batch["done"].unsqueeze(-1)

with torch.no_grad():
    next_act = act 
    q_next = q_net(next_obs, next_act)
    target_q = rew + gamma * (1 - done.float()) * q_next

q_val = q_net(obs, act)
loss = F.mse_loss(q_val, target_q)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")