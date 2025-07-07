import torch
from torch import nn, optim
from torch.distributions import Normal

import gym
from gym.wrappers import RecordVideo
import numpy as np

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

env = gym.make('Pendulum-v1', render_mode='rgb_array')
env = RecordVideo(env, 'ppo_pendulum_video', episode_trigger=lambda e:True)
t = lambda d: torch.tensor(d, device=device, dtype=torch.float32)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mu = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.fc(x)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        v = self.value(x)
        return mu, std, v
    
model = ActorCritic().to(device)
optimizer = optim.AdamW(model.parameters(), lr = 3e-4)

def get_action(state):
    state = t(state).unsqueeze(0)
    mu, std, v = model(state)
    dist = Normal(mu, std)
    a = dist.sample()
    logp = dist.log_prob(a).sum(1)
    return a.clamp(-2, 2).cpu().detach().numpy()[0], logp.detach(), v.detach()

def train():
    gamma = 0.99
    eps_clip = 0.2
    epoch = 10
    steps_per_epoch = 2048
    episode = 90
    
    for _ in tqdm(range(episode)):
        s_buf, a_buf, r_buf, logp_buf, v_buf = [], [], [], [], []
        s = env.reset()[0]
        
        for _ in range(steps_per_epoch):
            a, logp, v = get_action(s)
            
            s2, r, d, _, _ = env.step(a)
            s_buf.append(s)
            a_buf.append(a)
            r_buf.append(r)
            logp_buf.append(logp)
            v_buf.append(v)
            s = s2
            if d:
                s = env.reset()[0]
        
        v_buf = torch.cat(v_buf).squeeze().to(device)
        r_buf = np.array(r_buf)
        rets = []
        ret = 0
        
        for r in reversed(r_buf):
            ret = r + gamma * ret
            rets.insert(0, ret)
        rets = t(rets)
        
        advs = rets - v_buf
        s_buf = t(s_buf)
        a_buf = t(a_buf)
        logp_buf = torch.cat(logp_buf).to(device)
        
        for _ in range(epoch):
            
            mu, std, v = model(s_buf)
            dist = Normal(mu, std)
            logp = dist.log_prob(a_buf).sum(1)
            ratio = (logp - logp_buf).exp()
            
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advs
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = ((rets - v.squeeze()) ** 2).mean()
            
            loss = pi_loss + 0.5 * v_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    
    train()
    s = env.reset()[0]
    
    # test
    for _ in range(200):
        a, _, _ = get_action(s)
        s, _, d, _, _ = env.step(a)
        if d:
            break
        
    env.close()