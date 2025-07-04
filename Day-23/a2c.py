import gym
import torch
from torch import nn, optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        s = self.shared(x)
        l = self.actor(s)
        v = self.critic(s)
        return l, v
    
def train_one_episode(env:gym.Env, model, gamma):
    state, _ = env.reset()
    log_probs = []
    values = []
    rewards = []
    done = False
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_state, reward, done, _, _ = env.step(action.item())
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        
        state = next_state
    
    returns = compute_returns(rewards, gamma)
    return log_probs, values, returns, sum(rewards)

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)

def update(model, optimizer, log_probs, values, returns):
    log_probs = torch.stack(log_probs)
    values = torch.cat(values).squeeze()
    advantage = returns - values
    
    actor_loss = - (log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
    loss = actor_loss + critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def train(env, model, optimizer, episodes=250, gamma=0.99):
    
    for e in range(episodes):
        log_probs, values, returns, total_reward = train_one_episode(env, model, gamma)
        update(model, optimizer, log_probs, values, returns)
        
        print(f"Episode {e}, Total Reward: {total_reward}")
        
if __name__ == "__main__":
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    episodes = 250
    
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    train(env, model, optimizer, episodes)
    env.close()
    
        