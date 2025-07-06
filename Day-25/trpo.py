import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Normal
import numpy as np

env = gym.make("HalfCheetah-v4")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return mean, std

    def get_dist(self, obs):
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def get_log_prob(self, obs, act):
        dist = self.get_dist(obs)
        return dist.log_prob(act).sum(-1)

class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

def flat_grad(y, x, retain_graph=False):
    grad = torch.autograd.grad(y, x, create_graph=retain_graph)
    return torch.cat([g.view(-1) for g in grad])

def conjugate_gradients(Avp, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp_ = Avp(p)
        alpha = rdotr / torch.dot(p, Avp_)
        x += alpha * p
        r -= alpha * Avp_
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def fisher_vector_product(policy, obs, p, damping=1e-2):
    dist = policy.get_dist(obs)
    kl = torch.distributions.kl.kl_divergence(dist, dist.detach()).mean()
    kl_grad = flat_grad(kl, policy.parameters(), retain_graph=True)
    kl_p = (kl_grad * p).sum()
    kl_hess_p = flat_grad(kl_p, policy.parameters(), retain_graph=True)
    return kl_hess_p + damping * p

def linesearch(policy, obs, acts, advs, old_log_probs, fullstep, max_kl):
    max_backtracks = 10
    stepfrac = 1.0
    params = torch.cat([p.data.view(-1) for p in policy.parameters()])
    for _ in range(max_backtracks):
        new_params = params + stepfrac * fullstep
        set_params(policy, new_params)
        log_probs = policy.get_log_prob(obs, acts)
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate = (ratio * advs).mean()
        dist = policy.get_dist(obs)
        kl = torch.distributions.kl.kl_divergence(dist, dist.detach()).mean()
        if surrogate > 0 and kl < max_kl:
            return True
        stepfrac *= 0.5
    set_params(policy, params)
    return False

def set_params(model, flat_params):
    index = 0
    for p in model.parameters():
        p_length = p.numel()
        p.data.copy_(flat_params[index: index + p_length].view(p.size()))
        index += p_length

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

policy = Policy()
value_fn = Value()
obs = torch.tensor(env.reset(seed=42)[0], dtype=torch.float32).unsqueeze(0)
act = policy.get_dist(obs).sample()
log_prob = policy.get_log_prob(obs, act)
print("Observation:", obs)
print("Action:", act)
print("Log Prob:", log_prob)
