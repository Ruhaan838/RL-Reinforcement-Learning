import gym
import numpy as np
from collections import defaultdict

def td_zero(env_name="FrozenLake-v1", episodes=5000, lr=0.1, gamma=0.99):
    env = gym.make(env_name)
    V = defaultdict(float)
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()

            next_state, reward, done, truncated, _ = env.step(action)

            V[state] += lr * (reward + gamma * V[next_state] - V[state])
            state = next_state

    env.close()
    return V

value_table = td_zero(env_name="FrozenLake-v1")

for state, value in sorted(value_table.items()):
    print(f"V({state}) = {value:.3f}")
