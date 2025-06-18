from environment import TreasureTrapEnv
from eps_greedy import EpsGreedy

import numpy as np
from tqdm import tqdm

def train(env, policy, q_values, lr, episodes):
    for ep in (pbar := tqdm(range(episodes))):
        state = env.reset()
        action = policy.sel_action(q_values)
        _, reward, done = env.step(action)
        
        q_values[action] += lr * (reward - q_values[action])
        action_counts[action] += 1
        
        pbar.set_postfix_str(f"Episode {ep+1} - Q-Values:{q_values.round(2)}")
    

if __name__ == "__main__":
    
    env = TreasureTrapEnv()
    policy = EpsGreedy(n_actions=3, eps=0.1)
    q_values = np.zeros(3)
    action_counts = np.zeros(3)
    
    episodes = 100000
    lr = 0.1 
    
    train(env, policy, q_values, lr, episodes)
    
    print("\n Final action Values:")
    for i in range(3):
        print(f"Door {i}: Q = {q_values[i]:.2f}, chosen {int(action_counts[i])} times")
    
    