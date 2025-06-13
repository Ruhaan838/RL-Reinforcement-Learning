import numpy as np
from MDP import MDPModel

class ValueIterationAgent:
    def __init__(self, mdp: MDPModel, theta=1e-6):
        self.mdp = mdp
        self.theta = theta
        self.V = np.zeros(self.mdp.world.n_states)
        self.policy = np.zeros(self.mdp.world.n_states, dtype=int)

    def run(self):
        while True:
            delta = 0
            for s in range(self.mdp.world.n_states):
                if s == self.mdp.world.state_to_index(self.mdp.world.terminal_state):
                    continue
                q_values = np.zeros(self.mdp.world.n_actions)
                for a in range(self.mdp.world.n_actions):
                    q_values[a] = np.sum(
                        self.mdp.P[s, a] * (self.mdp.R[s, a] + self.mdp.gamma * self.V)
                    )
                max_q = np.max(q_values)
                delta = max(delta, np.abs(max_q - self.V[s]))
                self.V[s] = max_q
                self.policy[s] = np.argmax(q_values)
            if delta < self.theta:
                break

    def get_value_grid(self):
        return self.V.reshape(self.mdp.world.rows, self.mdp.world.cols)

    def get_policy_grid(self):
        return self.policy.reshape(self.mdp.world.rows, self.mdp.world.cols)
