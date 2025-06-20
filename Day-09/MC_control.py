import numpy as np
import random
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm

class AdClickEnv():
    def __init__(self, ad_click_probs):
        self.ad_click_probs = ad_click_probs
        self.num_ads = len(ad_click_probs)
        
    def reset(self):
        return 'user_lands'
    
    def step(self, action):
        clicked = np.random.rand() < self.ad_click_probs[action]
        reward = 1 if clicked else 0
        return 'user_lands', reward
    
def mc_control_exploring_starts(env, num_episodes=10000, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.num_ads))
    returns = defaultdict(lambda: [[] for _ in range(env.num_ads)])
    policy = {}
    
    for i in tqdm(range(num_episodes), desc="Episodes"):
        state = env.reset()
        
        action = random.randint(0, env.num_ads - 1)
        
        episode = []
        _, reward = env.step(action)
        episode.append((state, action, reward))
        
        G = 0
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t
            
            if not any((s == state_t and a == action_t) for s, a, r in episode[:t]):
                returns[state_t][action_t].append(G)
                Q[state_t][action_t] = np.mean(returns[state_t][action_t])
                
        for state in Q:
            policy[state] = np.argmax(Q[state])
    return policy, Q

ad_click_probs = [0.05, 0.12, 0.01, 0.04, 0.09]
env = AdClickEnv(ad_click_probs)

policy, Q = mc_control_exploring_starts(env)

for state in policy:
    print(f"State: {state} -> show ad #{policy[state]} (Estimated CTR: {Q[state][policy[state]]})")

ads = list(range(len(ad_click_probs)))
q_value = Q['user_lands']
plt.bar(ads, q_value)
plt.show()