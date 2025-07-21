import argparse
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import gymnasium as gym
except Exception:
    import gym


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    d: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: List[Transition] = []

    def add(self, t: Transition):
        if len(self.buf) >= self.capacity:
            self.buf.pop(0)
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32)
        a = torch.tensor([b.a for b in batch], dtype=torch.long)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.stack([b.s2 for b in batch]), dtype=torch.float32)
        d = torch.tensor([b.d for b in batch], dtype=torch.float32).unsqueeze(1)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out)
        )
    def forward(self, x):
        return self.net(x)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.mlp = MLP(obs_dim, n_actions)
    def forward(self, s):
        return self.mlp(s) 

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.mlp = MLP(obs_dim, n_actions)
    def forward(self, s):
        return self.mlp(s) 


def collect_dataset(env_name: str, episodes: int = 50, capacity: int = 50000, seed: int = 0) -> Tuple[ReplayBuffer, int, int]:
    set_seed(seed)
    env = gym.make(env_name)
    try:
        obs, _ = env.reset(seed=seed)
    except Exception:
        obs = env.reset()
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    buf = ReplayBuffer(capacity)

    for ep in range(episodes):
        try:
            s, _ = env.reset(seed=seed + ep)
        except Exception:
            s = env.reset()
        done = False
        while not done:
        
            a = env.action_space.sample()
            s2, r, term, trunc, _ = env.step(a) if hasattr(env, 'step') and len(env.step(a)) == 5 else (*env.step(a), False)
            d = bool(term or trunc) if 'trunc' in locals() else bool(term)
            buf.add(Transition(s, a, r, s2, d))
            s = s2
            done = d
    env.close()
    return buf, obs_dim, n_actions


def train_bc(buf: ReplayBuffer, obs_dim: int, n_actions: int, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, seed: int = 0) -> PolicyNet:
    set_seed(seed)
    policy = PolicyNet(obs_dim, n_actions)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        losses = []
        for _ in range(max(1, len(buf) // batch_size)):
            s, a, _, _, _ = buf.sample(batch_size)
            logits = policy(s)
            loss = F.cross_entropy(logits, a)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[BC] epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")
    return policy


def soft_update(target: nn.Module, src: nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), src.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data * tau)

@dataclass
class BCQConfig:
    gamma: float = 0.99
    tau: float = 0.3     
    lr_q: float = 1e-3
    lr_beta: float = 1e-3
    batch_size: int = 256
    steps: int = 20000
    target_update_tau: float = 0.01

class DiscreteBCQ:
    def __init__(self, obs_dim: int, n_actions: int, cfg: BCQConfig):
        self.beta = PolicyNet(obs_dim, n_actions)    
        self.q = QNet(obs_dim, n_actions)
        self.q_tgt = QNet(obs_dim, n_actions)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.cfg = cfg
        self.opt_q = optim.Adam(self.q.parameters(), lr=cfg.lr_q)
        self.opt_beta = optim.Adam(self.beta.parameters(), lr=cfg.lr_beta)

    def train_step(self, batch):
        s, a, r, s2, d = batch
    
        logits = self.beta(s)
        loss_beta = F.cross_entropy(logits, a)
        self.opt_beta.zero_grad(); loss_beta.backward(); self.opt_beta.step()

    
        with torch.no_grad():
            beta_probs = F.softmax(self.beta(s2), dim=1) 
            mask = (beta_probs > self.cfg.tau).float()   
            q_next = self.q_tgt(s2)
        
            has_any = (mask.sum(dim=1, keepdim=True) > 0).float()
            restricted = q_next - 1e9 * (1 - mask)       
            max_restricted, _ = restricted.max(dim=1, keepdim=True)
            max_full, _ = q_next.max(dim=1, keepdim=True)
            best_next = has_any * max_restricted + (1 - has_any) * max_full
            y = r + (1 - d) * self.cfg.gamma * best_next
        q_sa = self.q(s).gather(1, a.view(-1,1))
        loss_q = F.mse_loss(q_sa, y)
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        soft_update(self.q_tgt, self.q, self.cfg.target_update_tau)
        return loss_q.item(), loss_beta.item()

    @torch.no_grad()
    def act(self, s_np: np.ndarray) -> int:
        s = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0)
        beta_probs = F.softmax(self.beta(s), dim=1).squeeze(0)
        q_vals = self.q(s).squeeze(0)
        mask = (beta_probs > self.cfg.tau).float()
        if mask.sum().item() == 0:
            a = int(torch.argmax(q_vals).item())
        else:
            masked_q = q_vals - 1e9 * (1 - mask)
            a = int(torch.argmax(masked_q).item())
        return a


def evaluate_policy(env_name: str, policy_fn, episodes: int = 5, seed: int = 0) -> float:
    env = gym.make(env_name, render_mode=None)
    returns = []
    for ep in range(episodes):
        try:
            s, _ = env.reset(seed=seed + ep)
        except Exception:
            s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = policy_fn(s)
            step_out = env.step(a)
            if len(step_out) == 5:
                s, r, term, trunc, _ = step_out
                d = bool(term or trunc)
            else:
                s, r, term, _ = step_out
                d = bool(term)
            total += float(r)
            done = d
        returns.append(total)
    env.close()
    avg = float(np.mean(returns))
    print(f"Average return over {episodes} episodes: {avg:.2f}")
    return avg


def main():
    parser = argparse.ArgumentParser(description="Minimal Offline RL: BC + Discrete BCQ")
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_episodes', type=int, default=50)
    parser.add_argument('--capacity', type=int, default=50000)
    parser.add_argument('--algo', type=str, choices=['bc', 'bcq'], default='bc')
    parser.add_argument('--epochs', type=int, default=20, help='for BC')
    parser.add_argument('--steps', type=int, default=20000, help='for BCQ steps')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.3)
    args = parser.parse_args()

    buf, obs_dim, n_actions = collect_dataset(args.env, args.dataset_episodes, args.capacity, seed=args.seed)
    print(f"Dataset size: {len(buf)} transitions | obs_dim={obs_dim} | n_actions={n_actions}")

    if args.algo == 'bc':
        policy = train_bc(buf, obs_dim, n_actions, epochs=args.epochs, batch_size=args.batch, seed=args.seed)
        policy_fn = lambda s: int(torch.argmax(policy(torch.tensor(s, dtype=torch.float32))).item())
    else:
        cfg = BCQConfig(batch_size=args.batch, steps=args.steps, tau=args.tau)
        agent = DiscreteBCQ(obs_dim, n_actions, cfg)
        iters = 0
        while iters < cfg.steps:
            batch = buf.sample(cfg.batch_size)
            lq, lb = agent.train_step(batch)
            iters += 1
            if iters % 500 == 0:
                print(f"[BCQ] step {iters}/{cfg.steps}  q_loss={lq:.4f}  beta_loss={lb:.4f}")
        policy_fn = agent.act
        
    evaluate_policy(args.env, policy_fn, episodes=10, seed=args.seed)

if __name__ == '__main__':
    main()
