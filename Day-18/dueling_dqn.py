import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def select_action(model, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
        return q_values.argmax().item()


def train_step(model, target_model, optimizer, replay_buffer, batch_size, gamma):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    q_values = model(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_model(next_states).max(1, keepdim=True)[0]
        target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_dueling_dqn(env_name='MountainCar-v0',
                      episodes=300,
                      gamma=0.99,
                      lr=1e-3,
                      buffer_size=10000,
                      batch_size=64,
                      min_buffer_size=1000,
                      epsilon_start=1.0,
                      epsilon_end=0.02,
                      epsilon_decay_steps=20000,
                      target_update_freq=1000):

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DuelingDQN(state_dim, action_dim)
    target_net = DuelingDQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            total_steps += 1

            action = select_action(policy_net, state, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if epsilon > epsilon_end:
                epsilon -= epsilon_decay

            if len(replay_buffer) >= min_buffer_size:
                train_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma)

                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
        if episode % 10 == 0:
            print(f"Episode {episode} - Reward: {episode_reward:.2f} - Epsilon: {epsilon:.3f}")

    env.close()
if __name__ == '__main__':
    train_dueling_dqn(episodes=10000)
