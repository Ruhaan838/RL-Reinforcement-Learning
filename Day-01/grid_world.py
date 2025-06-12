import numpy as np

BORAD_ROWS = 3
BORAD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = True

class State:
    def __init__(self, state=START):
        self.board = np.zeros([BORAD_ROWS, BORAD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.detemine = DETERMINISTIC
        
    def give_Reward(self):
        if self.state == WIN_STATE:     return 1
        elif self.state == LOSE_STATE:  return -1
        else:                           return 0
    
    def is_end_func(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True
    
    def next_position(self, action:str):
        """
        action:str = "up", "down", "left", "right"
        """
        
        if self.detemine:
            if action == "up":
                next_state = (self.state[0] - 1, self.state[1])
            elif action == "down":
                next_state = (self.state[0] + 1, self.state[1])
            elif action == "left":
                next_state = (self.state[0], self.state[1] - 1)
            else:
                next_state = (self.state[0], self.state[1] + 1)
            
            if (next_state[0] >= 0 and next_state[0] <= 2):
                if (next_state[1] >= 0) and (next_state[1] <= 3):
                    if next_state != (1, 1):
                        return next_state
            return self.state
        
    def show_board(self):
        self.board[self.state] = 1
        for i in range(0, BORAD_ROWS):
            print('-'*10)
            out = '| '
            for j in range(0, BORAD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = 'o'
                out += token + ' | ' # type: ignore
            print(out)
        print('-'*10)


class Agent:
    def __init__(self, lr=0.2):
        self.state = []
        self.actions = ["up", "down", "left", "right"]
        self.state_cls = State()
        self.lr = lr
        self.exp_rate = 0.3
        
        self.state_vals = {}
        for i in range(BORAD_ROWS):
            for j in range(BORAD_COLS):
                self.state_vals[(i, j)] = 0
                
    def choose_action(self):
        max_next_reward = 0
        action = ""
        
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                next_reward = self.state_vals[self.state_cls.next_position(a)] 
                if next_reward >= max_next_reward:
                    action = a
                    max_next_reward = next_reward
        return action
                
    def take_action(self, action):
        pos = self.state_cls.next_position(action)
        return State(pos)
    
    def reset(self):
        self.state = []
        self.state_cls = State()
    
    def play(self, round=10):
        i = 0
        while i < round:
            if self.state_cls.isEnd: 
                reward = self.state_cls.give_Reward() 
                self.state_vals[self.state_cls.state] = reward 
                print("Game End: Reward", reward)
                for s in reversed(self.state):
                    reward = self.state_vals[s] + self.lr * (reward - self.state_vals[s])
                self.reset()
                i += 1
            else:
                action = self.choose_action()
                self.state.append(self.state_cls.next_position(action))
                print(f"current position {self.state_cls.state} action {action}")
                self.state_cls = self.take_action(action)
                
                self.state_cls.is_end_func()
                print("next State:", self.state_cls.state)
                print("-"*38)
                
    def show_values(self):
        for i in range(0, BORAD_ROWS):
            print('-'*38)
            out = '| '
            for j in range(0, BORAD_COLS):
                out += str(self.state_vals[(i, j)]).ljust(6) + ' | '
            print(out)
        print('-'*38)
    
if __name__ == "__main__":
    agent = Agent()
    agent.play(100)
    print(agent.show_values())
            
