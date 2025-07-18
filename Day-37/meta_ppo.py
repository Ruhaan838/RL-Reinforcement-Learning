import torch
import torch.nn.functional as F
from copy import deepcopy

def collect_trajectory(env, policy, steps, device):
    obs, _ = env.reset()
    traj = []
    for _ in range(steps):
        obs_tensor = torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = policy(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        next_obs, reward, done, _ = env.step(action)
        traj.append((obs_tensor, action, reward, value))
        obs = next_obs
        if done:
            break
    return traj

def compute_loss(traj, policy, gamma=0.99):
    returns, loss = 0, 0
    for obs, action, reward, value in reversed(traj):
        returns = reward + gamma * returns
        logits, value_pred = policy(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = log_probs[0, action]
        advantage = returns - value_pred
        policy_loss = -log_prob * advantage.detach()
        value_loss = F.mse_loss(value_pred, torch.tensor([[returns]]))
        loss += policy_loss + value_loss
    return loss
