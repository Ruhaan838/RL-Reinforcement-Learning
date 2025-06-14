import numpy as np

def policy_eval(env, policy, gamma=0.99, aplha=1e-6):
    #using_temporal_diffrence_method
    state_values = np.zeros(env.nS) # value funciton and env.nS is number of state
    init_state = env.reset() # inintalize the state by reseting the env

    for t in range(env.nS):
        action = policy(init_state)
        new_state, reward, done = env.step(action) 

        state_values[init_state] += aplha * (reward + gamma * state_values[new_state] - state_values[init_state])

        if done:
            state = env.reset()
        else:
            state = new_state

# we use the temporal diffrence method for policy evaluation 
# the formula for that is this given:
# v(s) <- v(s) + a(r_t + y * v(s_new) - v(s))
# for more info see readme