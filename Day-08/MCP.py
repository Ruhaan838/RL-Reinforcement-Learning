import numpy as np
from collections import defaultdict

def gen_episode():
    episode = []
    for state in range(4):
        next_state = state + 1
        reward = 1 if next_state == 4 else 0
        episode.append((state, reward))
    return episode

def monte_carlo_predict(policy_fn, episodes=10000):
    val_table = defaultdict(float)
    return_sum = defaultdict(float)
    return_count = defaultdict(int)
    
    for _ in range(episodes):
        episode = policy_fn()
        G = 0
        visited_state = set()
        
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + G
            
            if state not in visited_state:
                visited_state.add(state)
                return_sum[state] += G
                return_count[state] += 1
                val_table[state] = return_sum[state] / return_count[state]
                
    return val_table


V = monte_carlo_predict(gen_episode, episodes=5000)

for state in sorted(V):
    print(f"V({state}) = {V[state]:.3f}")

