import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ENV_NAME = "Pendulum-v1"
STATE_DIM = 3
ACTION_DIM = 1
LR = 3e-4
GAMMA = 0.99
CLIP_EPS = 0.2
UPDATE_EPOCHS = 10
STEPS_PER_UPDATE = 2048

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
t = lambda d:torch.tensor(d, dtype=torch.float32, device=device)

def get_obs(state):
    if isinstance(state, tuple):
        state = state[0]
    return np.array(state, dtype=np.float32)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(state)
        return mean, std, value

class Memory:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def store(self, state, action, logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

# ===== PPO Agent =====
class PPOAgent:
    def __init__(self):
        self.model = ActorCritic(STATE_DIM, ACTION_DIM).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.value_loss = nn.MSELoss().to(device)

    def select_action(self, state):
        state = t(state).unsqueeze(0)
        mean, std, value = self.model(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_clipped = torch.tanh(action) * 2.0  # Pendulum range is [-2, 2]
        logprob = dist.log_prob(action).sum(axis=-1)
        return action_clipped.detach().cpu().numpy()[0], logprob.item(), value.item()

    def update(self, memory):
        states = t(memory.states)
        actions = t(memory.actions)
        old_logprobs = t(memory.logprobs)
        rewards = t(memory.rewards)
        dones = t(memory.dones)
        values = t(memory.values)

        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + GAMMA * G * (1 - d)
            returns.insert(0, G)
        returns = t(returns)
        advantages = returns - values

        for _ in range(UPDATE_EPOCHS):
            mean, std, V = self.model(states)
            dist = torch.distributions.Normal(mean, std)
            logprobs = dist.log_prob(actions).sum(axis=-1)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_loss(V.squeeze(), returns)

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = PPOAgent()
    memory = Memory()

    state = get_obs(env.reset())
    total_steps = 0
    episode_reward = 0

    while True:

        for _ in range(STEPS_PER_UPDATE):
            action, logprob, value = agent.select_action(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            next_state = get_obs(next_state)
            memory.store(state, action, logprob, reward, done, value)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                print(f"Episode done | Reward: {episode_reward:.2f}")
                state = get_obs(env.reset())
                episode_reward = 0

        agent.update(memory)
        memory.clear()

        if total_steps >= 100_000:
            break

    torch.save(agent.model.state_dict(), "ppo_pendulum.pth")