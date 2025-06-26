import gym
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical

from tqdm import tqdm

env = gym.make('CartPole-v1')
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.softmax(self.l3(x), dim=-1)
    
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

def select_action(policy_net, state):
    state = torch.tensor(state, dtype=torch.float32, device=device)
    probs = policy_net(state)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)


policy_net = PolicyNet(4, 2).to(device)
value_net = ValueNet(4).to(device)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

def train(env, policy_net, value_net, policy_optimizer, value_optimizer, num_episodes=1000):
    state = env.reset()[0]
    log_probs, rewards, states = [], [], []
    
    done = False
    while not done:
        action, log_prob = select_action(policy_net, state)
        next_state, reward, done, _, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        states.append(torch.tensor(state, dtype=torch.float32))
        state = next_state
    
    returns = compute_returns(rewards).to(device)
    states = torch.stack(states).to(device)
    log_probs = torch.stack(log_probs)
    
    values = value_net(states).squeeze()
    advantages = returns - values.detach()
    
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    return policy_loss.item(), value_loss.item()

if __name__ == "__main__":
    
    for episode in (pbar := tqdm(range(1000))):
        policy_loss, value_loss = train(env, policy_net, value_net, policy_optimizer, value_optimizer)
        pbar.set_postfix({"policy_loss": policy_loss, "value_loss": value_loss})
    env.close()
        
    