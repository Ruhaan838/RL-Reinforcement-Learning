import gymnasium as gym 
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((state_space, action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)  
        else:
            return np.argmax(self.q_table[state])            

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class TaxiTrainer:
    def __init__(self, env_name="Taxi-v3", episodes=1000, max_steps=100):
        self.env = gym.make(env_name, render_mode=None)  
        self.agent = QLearningAgent(
            state_space=self.env.observation_space.n,
            action_space=self.env.action_space.n
        )
        self.episodes = episodes
        self.max_steps = max_steps

    def train(self):
        for episode in range(self.episodes):
            state, _ = self.env.reset() 
            done = False

            for _ in range(self.max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action) 
                done = terminated or truncated 
                
                self.agent.update_q_table(state, action, reward, next_state)
                state = next_state
                if done:
                    break

            self.agent.decay_epsilon()

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.episodes}, Epsilon: {self.agent.epsilon:.3f}")

        print("Training completed.\n")

    def evaluate(self, num_episodes=100):
        total_epochs, total_penalties = 0, 0

        for _ in range(num_episodes):
            state, _ = self.env.reset()  
            done = False
            epochs, penalties = 0, 0

            while not done:
                action = np.argmax(self.agent.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)  
                done = terminated or truncated  

                if reward == -10:
                    penalties += 1
                epochs += 1

            total_penalties += penalties
            total_epochs += epochs

        print(f"Results after {num_episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / num_episodes}")
        print(f"Average penalties per episode: {total_penalties / num_episodes}")

    def close(self):
        """Clean up the environment"""
        self.env.close()


if __name__ == "__main__":
    trainer = TaxiTrainer()
    try:
        trainer.train()
        trainer.evaluate()
    finally:
        trainer.close()  