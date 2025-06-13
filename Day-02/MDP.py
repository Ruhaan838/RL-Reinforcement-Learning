from grid_world import GridWorld
import numpy as np

class MDPModel:
    def __init__(self, world: GridWorld, gamma=0.9):
        self.world = world
        self.gamma = gamma
        self.P = self.build_transition_model()
        self.R = self.build_reward_model()

    def build_transition_model(self):
        P = np.zeros((self.world.n_states, self.world.n_actions, self.world.n_states))
        for s in range(self.world.n_states):
            state = self.world.index_to_state(s)
            for a in range(self.world.n_actions):
                next_state = self.world.get_next_state(state, a)
                s_prime = self.world.state_to_index(next_state)
                P[s, a, s_prime] = 1.0
        return P

    def build_reward_model(self):
        R = np.zeros((self.world.n_states, self.world.n_actions))
        goal_idx = self.world.state_to_index(self.world.terminal_state)
        for s in range(self.world.n_states):
            for a in range(self.world.n_actions):
                next_states = np.where(self.P[s, a] == 1.0)[0]
                if goal_idx in next_states:
                    R[s, a] = 1.0
        return R
