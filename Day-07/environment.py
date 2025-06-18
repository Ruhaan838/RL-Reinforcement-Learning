import numpy as np

class TreasureTrapEnv():
    def __init__(self):
        self.n_actions = 3
        
    def reset(self):
        return 0
    
    def step(self, action):
        if action == 0: reward = 0 #trap
        elif action == 1:
            reward = 1 if np.random.rand() < 0.7 else 0
        elif action == 2:
            reward = 1 #always treasure
        done = True #one step per episode
        return 0, reward, done #type:ignore