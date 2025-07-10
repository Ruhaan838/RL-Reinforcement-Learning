import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = lambda d: torch.tensor(d, dtype=torch.float32, device=device)

class CNNPPO_DQN(nn.Module):
    def __init__(self, action_dim):
        super(CNNPPO_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Linear(512, 1)
        self.dqn_head = nn.Linear(512, 5)  

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        mean = self.actor_mean(x)
        value = self.critic(x)
        q_values = self.dqn_head(x)
        return mean, self.actor_log_std, value, q_values

DISCRETE_ACTIONS = [
    np.array([-1.0, 0.0, 0.0]),  # steer left
    np.array([1.0, 0.0, 0.0]),   # steer right
    np.array([0.0, 1.0, 0.0]),   # throttle
    np.array([0.0, 0.0, 0.8]),   # brake
    np.array([0.0, 0.0, 0.0])    # do nothing
]

def preprocess(obs):
    obs = obs.transpose((2, 0, 1))  
    obs = obs / 255.0               
    return t(obs).unsqueeze(0)      

def train(env, model, optimizer, loss_fn, episodes=10):
    model.train()
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = preprocess(obs)
            mean, log_std, value, q_values = model(obs_tensor)

            std = torch.exp(log_std)
            action = mean + std * torch.randn_like(mean)
            action = torch.clamp(action, -1.0, 1.0)
            action_np = action.cpu().detach().numpy().flatten()

            q_idx = q_values.argmax(dim=-1).item()
            dqn_action_np = DISCRETE_ACTIONS[q_idx]

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            total_reward += reward

            target_value = t([reward])
            loss = loss_fn(value.squeeze(), target_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

        print(f"Episode {ep+1} Total Reward: {total_reward:.2f}")

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    action_dim = env.action_space.shape[0]
    model = CNNPPO_DQN(action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    train(env, model, optimizer, loss_fn, episodes=3)

    env.close()

if __name__ == "__main__":
    main()
