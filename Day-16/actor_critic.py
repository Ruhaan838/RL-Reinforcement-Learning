import gym
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.common = nn.Linear(state_dim, hidden_dim)
        
        self.actor = nn.Linear(hidden_dim, action_dim)   # policy
        self.critic = nn.Linear(hidden_dim, 1)           # value_func
    
    def forward(self, x):
        x = torch.tanh(self.common(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99

for episode in range(500):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = model(state_tensor)
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        next_state, reward, done, _, _ = env.step(action.item())
        
        # Value of next state
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_value = model(next_state_tensor)
        
        # Compute advantage(loss)
        td_target = reward + gamma * next_value * (1 - int(done))
        advantage = td_target - value
        
        loss = advantage.pow(2)
        
        # policy gradient loss
        actor_loss = -dist.log_prob(action) * advantage.detach()
        
        loss = actor_loss + loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode}: Total Reward = {total_reward}")
