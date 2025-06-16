import numpy as np

grid_size = 4 #help to make the grid env
n_states = grid_size * grid_size
n_actions = 4 #left, right, up, down
terminal_state = [0, n_states - 1] #indexing from 0 (fixed typo)
gamma = 0.9 #no discount
theta = 1e-5

action_map = {
    0:-grid_size, #up
    1: grid_size, #down
    2: -1, #left
    3: 1 #right
}

class Environment(): 
    def is_terminal(self, state): 
        return state in terminal_state
    
    def step(self, state, action):
        row, col = divmod(state, grid_size)
        
        #all possible where the wall exists
        if action == 0 and row == 0: return state 
        if action == 1 and row == grid_size - 1: return state
        if action == 2 and col == 0: return state
        if action == 3 and col == grid_size - 1: return state
        
        next_state = state + action_map[action]
        return next_state
    
    def reward(self, state, action, next_state):
        if next_state in terminal_state:
            return 0
        else:
            return -1  


class PolicyIteration():
    def __init__(self, env: Environment):
        self.env = env
        self.policy = np.zeros(n_states, dtype=int)
        self.value = np.zeros(n_states)
        
    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(n_states):
                if self.env.is_terminal(s):
                    continue
                a = self.policy[s]
                next_s = self.env.step(s, a)
                r = self.env.reward(s, a, next_s)
                new_v = r + gamma * self.value[next_s]
                delta = max(delta, abs(new_v - self.value[s]))
                self.value[s] = new_v
            if delta < theta:
                break
                
    def policy_improvement(self):
        policy_stable = True
        for s in range(n_states):
            if self.env.is_terminal(s):
                continue
            old_action = self.policy[s]
            action_values = []
            
            for a in range(n_actions):
                next_s = self.env.step(s, a)
                r = self.env.reward(s, a, next_s)
                action_values.append(r + gamma * self.value[next_s])
            
            best_action = np.argmax(action_values)
            self.policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable
    
    def run(self):
        iteration = 0
        while True:
            iteration += 1
            print(f"Policy Iteration {iteration}")
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print(f"Converged after {iteration} iterations!")
                break
            if iteration > 100: 
                print("Warning: Too many iterations, breaking...")
                break
            
    def print_policy(self):
        arrows = ['↑', '↓', '←', '→']
        grid = []
        for s, a in enumerate(self.policy):
            if s == 0:  # Start state
                grid.append('S')
            elif s == n_states - 1: 
                grid.append('E')
            elif s in terminal_state: 
                grid.append('.')
            else:
                grid.append(arrows[a])
        
        print("\nPolicy:")
        print(np.array(grid).reshape((grid_size, grid_size)))
        
    def print_values(self):
        print("Final State Values:")
        print(np.round(self.value.reshape((grid_size, grid_size)), 2))
        

if __name__ == "__main__":
    env = Environment()
    pi = PolicyIteration(env)
    pi.run() #run the policy iteration demo :)
    pi.print_values()
    pi.print_policy()