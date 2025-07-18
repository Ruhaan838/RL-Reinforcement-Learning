import gymnasium as gym
import random

TASKS = [
    'MiniGrid-Empty-5x5-v0',
    'MiniGrid-DoorKey-5x5-v0',
    'MiniGrid-MultiRoom-N2-S4-v0',
    'MiniGrid-Empty-Random-6x6-v0'
]

def sample_task(seed=None):
    env_id = random.choice(TASKS)
    env = gym.make(env_id)
    if seed is not None:
        env.seed(seed)
    return env
