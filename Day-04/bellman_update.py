import numpy as np

n_states = 4
n_actions = 2
gamma = 0.9
theta = 1e-6

# translation matrix: P[state][action] = (prob, next_state, reward)
P = {
    0: {
        0:[(1.0, 0, 0)],
        1:[(1.0, 1, 0)],
    },
    1: {
        0:[(1.0, 0, 0)],
        1:[(1.0, 2, 0)],
    },
    2: {
        0:[(1.0, 1, 0)],
        1:[(1.0, 3, 1)],
    },
    3: {
        0:[(1.0, 3, 0)],
        1:[(1.0, 3, 0)],
    },
}

def value_iter(P, n_states, n_actions, gamma=0.9, theta=1e-6):
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            action_value = []
            for a in range(n_actions):
                total = 0
                for prob, next_s, reward in P[s][a]:
                    total += prob * (reward + gamma * V[next_s])
                action_value.append(total)
            V[s] = max(action_value) #get the max action
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    ## bellman 
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        action_value = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_s, reward in P[s][a]:
                action_value[a] += prob * (reward + gamma * V[next_s])
        
        best_action = np.argmax(action_value)
        policy[s][best_action] = 1.0
        
    return V, policy

if __name__ == "__main__":
    V, policy = value_iter(P, n_states, n_actions)
    print("Optimal Action\n", V)
    print("\nOptimal Policy", policy)
    