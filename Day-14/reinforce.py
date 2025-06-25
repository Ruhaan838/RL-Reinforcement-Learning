import gym
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 128)
        self.l2 = nn.Linear(128, act_dim)
    def forward(self, x):
        x = F.relu(self.l1(x))
        out = self.l2(x)
        return F.softmax(out, dim=-1)
    
class ReinforceAgent():
    def __init__(self, obs_dim, act_dim, lr=1e-2, gamma=0.99):
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optim = optim.AdamW(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, obs):
        state = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update_policy(self):
        G = 0
        returns = []
        
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = 0
        for log_prob, Gt in zip(self.log_probs, returns):
            loss -= log_prob * Gt
            
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        self.log_probs = []
        self.rewards = []
        
def train_reinforce(env_name='CartPole-v1', num_episodes=1000):
    env = gym.make(env_name)
    agent = ReinforceAgent(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n)
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle new Gym API
            state = state[0]
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            if isinstance(info, dict):  # Handle new Gym API
                truncated = info.get('truncated', False)
                done = info.get('terminated', done)
            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        agent.update_policy()
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train_reinforce(num_episodes=100)