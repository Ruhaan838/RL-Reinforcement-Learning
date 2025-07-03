import torch
from torch import optim
from torch.nn import functional as F
import numpy as np
import random
from tqdm import tqdm

from models import DQN
from buffer import ReplayBuffer

class Agent:
    def __init__(self, env, double=False, dueling=False):
        self.env = env
        self.double = double
        self.dueling = dueling
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.model = DQN((4, 84, 84), env.action_space.n, dueling).to(self.device)
        self.target = DQN((4, 84, 84), env.action_space.n, dueling).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        
        self.optim = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(100000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.1
        
    def preprocess(self, obs):
        obs = obs[32:195]  # crop
        obs = obs[::2, ::2, 0]  # downsample by factor of 2, take R channel
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1
        # pad to (84, 84) if needed
        h, w = obs.shape
        pad_h = 84 - h
        pad_w = 84 - w
        if pad_h > 0 or pad_w > 0:
            obs = np.pad(obs, ((0, pad_h), (0, pad_w)), mode='constant')
        return obs.astype(np.float32)
    
    def stack_frames(self, frames):
        return np.stack(frames, axis=0)
    
    def select_action(self, state):
        if random.random() < self.eps:
            return self.env.action_space.sample()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax(1).item()
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        
        tensor = lambda data, dtype=torch.float32: torch.tensor(data, dtype=dtype, device=self.device)
        
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        
        state = tensor(state)
        next_state = tensor(next_state)
        action = tensor(action, torch.int64).unsqueeze(1)  # shape [batch, 1]
        reward = tensor(reward)
        done = tensor(done)

        q_values = self.model(state).gather(1, action)
        
        with torch.no_grad():
            if self.double:
                next_action = self.model(next_state).argmax(1, keepdim=True)
                next_q = self.target(next_state).gather(1, next_action)
            else:
                next_q = self.target(next_state).max(1, keepdim=True)[0]
            exp_q = reward + self.gamma * next_q * (1 - done)
        
        loss = F.mse_loss(q_values, exp_q)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
        
    def train(self, episodes):
        for e in (pbar := tqdm(range(episodes))):
            obs, _ = self.env.reset()
            frame = self.preprocess(obs)
            frames = [frame] * 4  
            state = self.stack_frames(frames)
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_obs, reward, done, _, _ = self.env.step(action)
                next_frame = self.preprocess(next_obs)
                frames.pop(0)
                frames.append(next_frame)
                next_state = self.stack_frames(frames)
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.train_step()
            self.eps = max(self.eps * self.eps_decay, self.eps_min)

            self.update_target()

            pbar.set_postfix(Episodes=episodes, Reward=total_reward)