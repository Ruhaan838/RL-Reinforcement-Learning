import numpy as np

class ValueIteration:
    def __init__(self, n_states, n_actions, transition_probabilities, rewards, gamma=0.9, theta=1e-5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transition_probabilities  
        self.R = rewards                   
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)

    def run(self):
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.V[s]
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    action_values[a] = self.R[s, a] + self.gamma * np.sum(self.P[s, a] * self.V)
                self.V[s] = np.max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break
        self.extract_policy()

    def extract_policy(self):
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                action_values[a] = self.R[s, a] + self.gamma * np.sum(self.P[s, a] * self.V)
            self.policy[s] = np.argmax(action_values)

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.V


n_states = 4
n_actions = 2
P = np.array([
    [[0.7, 0.3, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0]],
    [[0.0, 0.6, 0.4, 0.0], [0.0, 0.3, 0.7, 0.0]],
    [[0.0, 0.0, 0.8, 0.2], [0.0, 0.0, 0.5, 0.5]],
    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
])
R = np.array([
    [5, 10],
    [0, 0],
    [0, 0],
    [0, 0]
])

vi = ValueIteration(n_states, n_actions, P, R)
vi.run()
print("Optimal Value Function:", vi.get_value_function())
print("Optimal Policy:", vi.get_policy())
