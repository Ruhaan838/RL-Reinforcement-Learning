import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import gymnasium_robotics
from tabulate import tabulate
import pandas as pd

class HerBuf:
    def __init__(self, size, obs_dim, act_dim, goal_dim, her_k=4, eps=0.05):
        self.size = size
        self.ptr = 0
        self.full = False
        self.her_k = her_k
        self.eps = eps

        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

        self.ag = np.zeros((size, goal_dim), dtype=np.float32)
        self.g = np.zeros((size, goal_dim), dtype=np.float32)

        self.ep_idx = []
        self.cur = []
        self.idx_to_ep = {}

    def start_ep(self):
        self.cur = []

    def store(self, o, a, r, nx, d, ag, g):
        i = self.ptr
        self.obs[i] = o
        self.next_obs[i] = nx
        self.act[i] = a
        self.rew[i] = r
        self.done[i] = d
        self.ag[i] = ag
        self.g[i] = g
        self.cur.append(i)
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def end_ep(self):
        ep = np.array(self.cur, dtype=np.int32)
        self.ep_idx.append(ep)
        for idx in ep:
            self.idx_to_ep[idx] = ep
        self.cur = []
        if len(self.ep_idx) > 10000:
            self.ep_idx = self.ep_idx[-10000:]

    def _r(self, ag, g):
        return (np.linalg.norm(ag - g, axis=1) > self.eps).astype(np.float32) * -1.0

    def sample(self, batch):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch)

        o = self.obs[idx]
        a = self.act[idx]
        r = self.rew[idx]
        nx = self.next_obs[idx]
        d = self.done[idx]
        ag = self.ag[idx]
        g = self.g[idx]

        if self.ep_idx and self.her_k > 0:
            mask = np.random.rand(batch) < (self.her_k / (self.her_k + 1))
            her_idxs = np.where(mask)[0]

            valid_her_idxs = [j for j in her_idxs if idx[j] in self.idx_to_ep]

            if valid_her_idxs:
                ep_array = [self.idx_to_ep[idx[j]] for j in valid_her_idxs]
                pos_array = [np.where(ep == idx[j])[0][0] for ep, j in zip(ep_array, valid_her_idxs)]

                future_idx = []
                for ep, pos in zip(ep_array, pos_array):
                    if pos < len(ep) - 1:
                        future_idx.append(np.random.choice(ep[pos+1:]))
                    else:
                        future_idx.append(-1)

                future_idx = np.array(future_idx)
                valid_mask = future_idx >= 0
                if np.any(valid_mask):
                    f_ids = future_idx[valid_mask]
                    new_g = self.ag[f_ids]
                    gi = np.array(valid_her_idxs)[valid_mask]
                    g[gi] = new_g
                    r[gi] = self._r(ag[gi], new_g)[:, None]
                    o[gi, -new_g.shape[1]:] = new_g
                    nx[gi, -new_g.shape[1]:] = new_g
                    d[gi] = (r[gi] == 0.0).astype(np.float32)

        return (
            torch.as_tensor(o, dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.float32),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(nx, dtype=torch.float32),
            torch.as_tensor(d, dtype=torch.float32)
        )


class Net(nn.Module):
    def __init__(self, inp, out, act=False):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out)
        )
        self.act = act

    def forward(self, x):
        y = self.m(x)
        if self.act:
            y = torch.tanh(y)
        return y


if __name__ == "__main__":

    env = gym.make("FetchReach-v3")
    obs_sample = env.reset()[0]

    obs_dim = obs_sample["observation"].shape[0] + obs_sample["desired_goal"].shape[0]
    goal_dim = obs_sample["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]

    buf = HerBuf(200000, obs_dim, act_dim, goal_dim, her_k=4, eps=0.05)

    pi = Net(obs_dim, act_dim, act=True)
    q = Net(obs_dim + act_dim, 1)
    qt = Net(obs_dim + act_dim, 1)
    qt.load_state_dict(q.state_dict())

    pi_t = Net(obs_dim, act_dim, act=True)
    pi_t.load_state_dict(pi.state_dict())

    opt_pi = optim.Adam(pi.parameters(), lr=1e-3)
    opt_q = optim.Adam(q.parameters(), lr=1e-3)

    gamma = 0.98
    tau = 0.005
    batch = 256
    steps = 50000
    warmup = 2000

    s_dict, _ = env.reset()
    s = np.concatenate([s_dict["observation"], s_dict["desired_goal"]]).astype(np.float32)

    for t in range(steps):
        with torch.no_grad():
            if t < warmup:
                a = np.random.uniform(-1, 1, size=act_dim).astype(np.float32)
            else:
                aa = pi(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                noise = np.random.normal(0, 0.2, size=act_dim).astype(np.float32)
                a = np.clip(aa + noise, -1, 1).astype(np.float32)

        ns_dict, r, d, tr, info = env.step(a)
        ns = np.concatenate([ns_dict["observation"], ns_dict["desired_goal"]]).astype(np.float32)

        buf.store(s, a, r, ns, float(d), ns_dict["achieved_goal"], ns_dict["desired_goal"])
        s = ns

        if d or tr:
            buf.end_ep()
            s_dict, _ = env.reset()
            s = np.concatenate([s_dict["observation"], s_dict["desired_goal"]]).astype(np.float32)

        if t >= warmup:
            o, a, r, nx, dn = buf.sample(batch)

            with torch.no_grad():
                at = pi_t(nx)
                qtarg = qt(torch.cat([nx, at], dim=-1))
                y = r + gamma * (1.0 - dn) * qtarg

            qv = q(torch.cat([o, a], dim=-1))
            lq = (qv - y).pow(2).mean()

            opt_q.zero_grad()
            lq.backward()
            opt_q.step()

            for p in q.parameters():
                p.requires_grad = False

            ap = pi(o)
            lp = -q(torch.cat([o, ap], dim=-1)).mean()

            opt_pi.zero_grad()
            lp.backward()
            opt_pi.step()

            for p in q.parameters():
                p.requires_grad = True

            with torch.no_grad():
                for p, pt in zip(q.parameters(), qt.parameters()):
                    pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
                for p, pt in zip(pi.parameters(), pi_t.parameters()):
                    pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
                    
        metrics_log = []

        if (t + 1) % 5000 == 0:
            metrics_log.append({
                "Step": t + 1,
                "Q Loss": float(lq.item()),
                "Policy Loss": float(lp.item()),
                "Avg Reward": float(r.mean().item()),
                "Min Reward": float(r.min().item()),
                "Max Reward": float(r.max().item())
            })

            df = pd.DataFrame(metrics_log)
            print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
