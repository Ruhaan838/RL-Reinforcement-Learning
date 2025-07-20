import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt  # <-- add import

class StateEmbedding(nn.Module):
    def __init__(self, obs_dim, embed_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, embed_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)

class Manager(nn.Module):
    def __init__(self, obs_dim, goal_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, goal_dim)

    def forward(self, obs):
        return F.normalize(self.fc(obs), dim=-1)

class Worker(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return self.fc(x)

def compute_intrinsic_reward(goal, f_s, f_s_next):
    delta = f_s_next - f_s
    delta = F.normalize(delta, dim=-1)
    return F.cosine_similarity(goal, delta, dim=-1)

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
goal_dim = 8

f = StateEmbedding(obs_dim, goal_dim)
manager = Manager(obs_dim, goal_dim)
worker = Worker(obs_dim, goal_dim, act_dim)

num_steps = 100
intr_rewards = []

obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
for _ in range(num_steps):
    goal = manager(obs)
    f_s = f(obs)
    action_logits = worker(obs, goal)
    action = torch.argmax(action_logits, dim=-1).item()
    next_obs, reward, done, truncated, _ = env.step(action)
    f_s_next = f(torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0))
    intr_reward = compute_intrinsic_reward(goal, f_s, f_s_next)
    intr_rewards.append(intr_reward.item())
    obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
    if done or truncated:
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

plt.plot(intr_rewards)
plt.xlabel("Step")
plt.ylabel("Intrinsic Reward")
plt.title("Intrinsic Reward over Time")
plt.show()
