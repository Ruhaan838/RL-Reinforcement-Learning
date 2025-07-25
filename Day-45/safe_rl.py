import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

# Constraint: pole angle should not exceed ±0.15 rad on average
MAX_ALLOWED_ANGLE = 0.15

def constrained_rl_episode():
    obs, _ = env.reset()
    total_reward = 0
    total_cost = 0
    done = False

    while not done:
        action = env.action_space.sample()  # Random policy
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        angle = obs[2]  # Pole angle
        cost = 1.0 if abs(angle) > MAX_ALLOWED_ANGLE else 0.0

        total_reward += reward
        total_cost += cost

    return total_reward, total_cost

for ep in range(5):
    R, C = constrained_rl_episode()
    print(f"ep {ep+1}: reward={R}, cost={C}")
