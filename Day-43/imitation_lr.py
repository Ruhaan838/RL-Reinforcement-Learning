import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

env = gym.make("CartPole-v1")

expert_states = []
expert_actions = []

obs, _ = env.reset()
for _ in range(5000):
    action = env.action_space.sample()
    expert_states.append(obs)  # pure NumPy array
    expert_actions.append(action)
    obs, _, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

expert_states = np.array(expert_states, dtype=np.float32)
expert_actions = np.array(expert_actions, dtype=np.int64)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.fc(x)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
model = PolicyNet(obs_dim, act_dim)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

states_tensor = torch.tensor(expert_states)
actions_tensor = torch.tensor(expert_actions)

for epoch in range(10):
    logits = model(states_tensor)
    loss = criterion(logits, actions_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

obs, _ = env.reset()
total_reward = 0
for _ in range(200):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action = model(obs_tensor).argmax().item()
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    if done or truncated:
        break

print("Total reward with BC policy:", total_reward)
env.close()
