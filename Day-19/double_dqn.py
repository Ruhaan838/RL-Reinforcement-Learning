import gym
import numpy as np
import random
from collections import deque
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.layer3(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, action, rewards, next_reward, dones = zip(*batch)
        return np.array(states), action, rewards, np.array(next_reward), dones
    
    def __len__(self):
        return len(self.buffer)

env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

buffer_cap = 50_000
batch_size = 64
gamma = 0.99
lr = 1e-3
target_update_freq = 100
eps_start = 1.0
eps_end = 0.01
eps_decay = 500

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

online_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(online_net.state_dict())

loss_fn = nn.MSELoss().to(device)
optimizer = optim.Adam(online_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_cap)

num_episodes = 500
steps_done = 0

def train(env, batch_size, gamma, 
          eps_start, eps_end, eps_decay, 
          device, online_net, target_net, 
          loss_fn, optimizer, replay_buffer, 
          num_episodes, steps_done):
    
    
    for episodes in (pbar := tqdm(range(num_episodes))):
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        episodes_reward = 0
    
        done = False
    
        while not done:
            eps = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
            steps_done += 1
        
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = online_net(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done, _, _ = env.step(action) if len(env.step(action)) == 5 else (*env.step(action), None)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episodes_reward += reward
        
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
                
                q_values = online_net(states).gather(1, actions).squeeze(1)
            
                next_q_online = online_net(next_states)
                next_action = next_q_online.argmax(1, keepdim=True)
                next_q_target = target_net(next_states)
                next_q_values = next_q_target.gather(1, next_action).squeeze(1)
            
                exp_q_values = rewards + gamma * next_q_values * (1 - dones)
            
                loss = loss_fn(q_values, exp_q_values)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        pbar.set_postfix(Episodes=episodes, Reward=episodes_reward, Eps=eps)

def test(env, device, online_net):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = online_net(state_tensor)
            action = q_values.argmax().item()
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward

        print(f"Test Reward: {total_reward}")
        env.close()


train(env, batch_size, gamma, 
      eps_start, eps_end, eps_decay, 
      device, online_net, target_net, 
      loss_fn, optimizer, replay_buffer, 
      num_episodes, steps_done)

test(env, device, online_net)