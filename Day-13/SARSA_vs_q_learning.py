import gym
import numpy as np
import matplotlib.pyplot as plt

def train_taxi(method='q_learning', episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, render=False):
    
    env = gym.make("Taxi-v3")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for ep in range(episodes):
        # Handle Gym reset API change
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        done = False
        total_reward = 0
        
        # Initial action (needed for SARSA)
        if method == 'sarsa':
            action = (
                np.random.randint(n_actions)
                if np.random.rand() < epsilon
                else np.argmax(Q[state])
            )

        while not done:
            if render and ep % 1000 == 0:
                env.render()

            # Choose action (Q-Learning chooses inside loop)
            if method == 'q_learning':
                action = (
                    np.random.randint(n_actions)
                    if np.random.rand() < epsilon
                    else np.argmax(Q[state])
                )

            # Handle Gym step API change
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            total_reward += reward

            if method == 'sarsa':
                # Choose next action using ε-greedy policy
                next_action = (
                    np.random.randint(n_actions)
                    if np.random.rand() < epsilon
                    else np.argmax(Q[next_state])
                )
                # SARSA Update
                target = reward + gamma * Q[next_state][next_action]
            elif method == 'q_learning':
                # Q-Learning Update
                target = reward + gamma * np.max(Q[next_state])
            else:
                raise ValueError("Method must be 'sarsa' or 'q_learning'.")

            # Q update for both
            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state
            if method == 'sarsa':
                action = next_action  # For SARSA: continue with next action

        rewards.append(total_reward)

    env.close()
    print(f"Training complete using: {method.upper()}")
    return Q, rewards

# Train using SARSA
Q_sarsa, rewards_sarsa = train_taxi('sarsa')

# Train using Q-learning
Q_q, rewards_q = train_taxi('q_learning')

# Plotting
plt.plot(rewards_sarsa, label="SARSA")
plt.plot(rewards_q, label="Q-Learning")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SARSA vs Q-Learning")
plt.legend()
plt.grid()
plt.show()
