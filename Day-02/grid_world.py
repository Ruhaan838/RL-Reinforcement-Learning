class GridWorld:
    def __init__(self, rows=3, cols=3, terminal_state=(2, 2)):
        self.rows = rows
        self.cols = cols
        self.terminal_state = terminal_state
        self.n_states = rows * cols
        self.n_actions = 4  # up, down, left, right
        self.actions = ['up', 'down', 'left', 'right']
    
    def state_to_index(self, state):
        return state[0] * self.cols + state[1]

    def index_to_state(self, index):
        return (index // self.cols, index % self.cols)

    def is_terminal(self, state):
        return state == self.terminal_state

    def get_next_state(self, state, action):
        i, j = state
        if self.is_terminal(state):
            return state
        if action == 0:  # up
            i = max(i - 1, 0)
        elif action == 1:  # down
            i = min(i + 1, self.rows - 1)
        elif action == 2:  # left
            j = max(j - 1, 0)
        elif action == 3:  # right
            j = min(j + 1, self.cols - 1)
        return (i, j)