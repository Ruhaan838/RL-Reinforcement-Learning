import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import multiprocessing as mp
from torch.distributions import Categorical

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gym

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        p_l = self.policy(x)
        v = self.value(x)
        return p_l, v
    
def worker(model, optimizer, env:gym.Env, ep, max_eps):
    
    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    local_model.load_state_dict(model.state_dict())
    
    gamma = 0.99
    
    while ep.value < max_eps:
        state, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        
        while not done:
            state_tensor = torch.from_numpy(state).float()
            logits, value = local_model(state_tensor)
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done, _, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                Q_val = 0
                if not done:
                    next_state_tensor = torch.from_numpy(next_state).float()
                    _, Q_val = local_model(next_state_tensor)
                    
                    Q_vals = []
                    for r in reversed(rewards):
                        Q_val = r + gamma * Q_val
                        Q_vals.insert(0, Q_val)
                        
                    log_probs = torch.stack(log_probs)
                    values = torch.stack(values).squeeze()
                    Q_vals = torch.tensor(Q_vals)
                    
                    advantage = Q_vals - values
                    
                    actor_loss = -(log_probs * advantage.detach()).mean()
                    critic_loss = advantage.pow(2).mean()
                    loss = actor_loss + 0.5 * critic_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    for local_param, param in zip(local_model.parameters(), model.parameters()):
                        param._grad = local_param.grad
                    
                    optimizer.step()
                    local_model.load_state_dict(model.state_dict())
                    
                    with ep.get_lock():
                        ep.value += 1
                        print(f"\rEpisode {ep.value}/{max_eps} | Total Reward: {sum(rewards):.2f}", end="")
                    
                    break
                
if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = ActorCritic(input_dim, action_dim)
    model.share_memory()
    optimzer = optim.AdamW(model.parameters(), lr=1e-4)
    ep = mp.Value('i', 0)
    max_eps = 500
    
    processes = []
    num_workers = mp.cpu_count()
    
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(model, optimzer, env, ep, max_eps))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
