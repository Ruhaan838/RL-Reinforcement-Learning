import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import sys
import numpy as np


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

env = gym.make("Hopper-v5")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

actor = Actor().to(device)
critic = Critic().to(device)
actor_old = Actor().to(device)
actor_old.load_state_dict(actor.state_dict())

critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
def get_action(state):
    state = torch.from_numpy(np.array(state)).float().to(device)
    mean, std = actor(state)
    dist = Normal(mean, std)
    action = dist.sample()
    return action.cpu().detach().numpy(), dist.log_prob(action).sum()

def compute_advantages(rewards, values, dones, gamma=0.99, lam=0.97):
    adv = []
    gae = 0
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        adv.insert(0, gae)
    return adv

def ppo_update(states, actions, old_log_probs, returns, advantages, clip_ratio=0.2):
    for _ in range(10):
        mean, std = actor(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        ratio = torch.exp(log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        loss = -torch.min(ratio * advantages, clip_adv).mean()
        actor_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()

def flat_grad(y, model, retain_graph=False):
    grads = torch.autograd.grad(y, model.parameters(), retain_graph=retain_graph)
    return torch.cat([g.view(-1) for g in grads])

def flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def set_params(model, flat_params):
    idx = 0
    for param in model.parameters():
        size = param.numel()
        param.data.copy_(flat_params[idx:idx+size].view_as(param))
        idx += size

def trpo_step(states, actions, old_log_probs, advantages, max_kl=1e-2, cg_iters=10, damping=1e-2):
    mean, std = actor(states)
    dist = Normal(mean, std)
    log_probs = dist.log_prob(actions).sum(axis=-1)
    ratio = torch.exp(log_probs - old_log_probs)
    loss = -(ratio * advantages).mean()

    grads = flat_grad(loss, actor, retain_graph=True)

    def Fvp(v):
        mean, std = actor(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        kl = torch.distributions.kl_divergence(dist, Normal(*actor_old(states))).mean()
        kl_grad = flat_grad(kl, actor, create_graph=True)
        kl_v = (kl_grad * v).sum()
        hvp = flat_grad(kl_v, actor, retain_graph=True).detach()
        return hvp + damping * v

    x = torch.zeros_like(grads)
    r = grads.clone()
    p = grads.clone()
    for _ in range(cg_iters):
        Avp = Fvp(p)
        alpha = r.dot(r) / p.dot(Avp)
        x += alpha * p
        r_new = r - alpha * Avp
        beta = r_new.dot(r_new) / r.dot(r)
        p = r_new + beta * p
        r = r_new

    step_dir = x
    shs = 0.5 * step_dir.dot(Fvp(step_dir))
    lm = torch.sqrt(shs / max_kl)
    full_step = step_dir / lm
    old_params = flat_params(actor)
    set_params(actor, old_params + full_step)

arg = sys.argv
num_episodes = 1000
ppo = bool(arg[0])


for episode in range(num_episodes):
    state, _ = env.reset()
    states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
    done = False

    while not done:
        action, log_prob = get_action(state)
        value = critic(torch.from_numpy(np.array(state)).float().to(device)).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        dones.append(done)
        values.append(value)
        state = next_state

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)

    states = torch.FloatTensor(states).to(device)
    actions = torch.FloatTensor(actions).to(device)
    old_log_probs = torch.stack(log_probs).detach().to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = compute_advantages(rewards, values, dones)
    advantages = torch.FloatTensor(advantages).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(10):
        value_loss = ((critic(states) - returns)**2).mean()
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

    actor_old.load_state_dict(actor.state_dict())

    if ppo:
        ppo_update(states, actions, old_log_probs, returns, advantages)
    else:
        trpo_step(states, actions, old_log_probs, advantages)

    print(f"Episode: {episode}, Return: {sum(rewards):.2f}")
