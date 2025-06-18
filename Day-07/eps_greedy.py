import numpy as np

class EpsGreedy:
    def __init__(self, n_actions, eps):
        self.n_actions = n_actions
        self.eps = eps
        
    def sel_action(self, q_values):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(q_values)