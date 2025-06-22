import gym
import numpy as np
import random
from collections import defaultdict

env = gym.make("FrozenLake-v1", is_slippery=True)

lr = 0.1       
gamma = 0.99      
epsilon = 0.1     
episodes = 5000

Q = defaultdict(lambda: np.zeros(env.action_space.n))

def epsilon_greedy_policy(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

for episode in range(episodes):
    state = env.reset()[0]
    action = epsilon_greedy_policy(state)

    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = epsilon_greedy_policy(next_state)

        td_target = reward + gamma * Q[next_state][next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += lr * td_error

        state = next_state
        action = next_action
policy = {state: np.argmax(actions) for state, actions in Q.items()}

def evaluate_policy(env, policy, episodes=100):
    wins = 0
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = policy.get(state, env.action_space.sample())
            state, reward, done, _, _ = env.step(action)
            if done and reward == 1.0:
                wins += 1
    return wins / episodes

print(f"Policy success rate: {evaluate_policy(env, policy) * 100:.2f}%")
