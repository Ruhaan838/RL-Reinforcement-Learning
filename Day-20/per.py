import gym
import numpy as np
import torch
from torch import nn, optim
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


def one_hot(state, state_space):
    vec = np.zeros(state_space, dtype=np.float32)
    vec[state] = 1.0
    return vec


def train(env:gym.Env, q_net, target_q_net, optimizer, replay_buffer, num_episodes=500):
    state_space = env.observation_space.n
    action_space = env.action_space.n

    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    beta = 0.4
    beta_increment = 1e-4

    for episode in range(num_episodes):
        state, _= env.reset()
        state = one_hot(state, state_space)
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state_one_hot = one_hot(next_state, state_space)

            replay_buffer.push(state, action, reward, next_state_one_hot, done)
            state = next_state_one_hot
            total_reward += reward

            if len(replay_buffer.buffer) > batch_size:
                beta = min(1.0, beta + beta_increment)

                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                weights = torch.FloatTensor(weights)

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_q_net(next_states).max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                td_errors = q_values - expected_q_values.detach()
                loss = (weights * td_errors.pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
                replay_buffer.update_priorities(indices, new_priorities)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 20 == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode} | Total Reward: {total_reward} | Epsilon: {epsilon:.3f}")

    target_q_net.load_state_dict(q_net.state_dict())


def test(env, q_net, num_episodes=10, render=False):
    state_space = env.observation_space.n

    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = one_hot(state, state_space)
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            state = one_hot(next_state, state_space)
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode} | Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average test reward over {num_episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    state_space = env.observation_space.n
    action_space = env.action_space.n

    q_net = QNetwork(state_space, action_space)
    target_q_net = QNetwork(state_space, action_space)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = PrioritizedReplayBuffer(10000)

    train(env, q_net, target_q_net, optimizer, replay_buffer, num_episodes=500)
    test(env, q_net, num_episodes=10, render=False)

    env.close()
