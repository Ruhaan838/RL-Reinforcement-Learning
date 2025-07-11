import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

class C51(nn.Module):
    def __init__(self, state_dim, action_dim, atom_size, v_min, v_max):
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        self.support = torch.linspace(v_min, v_max, atom_size)

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim * atom_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.action_dim, self.atom_size)
        prob = F.softmax(x, dim=2)
        return prob

    def q_values(self, x):
        prob = self.forward(x)
        q = torch.sum(prob * self.support, dim=2)
        return q

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    atom_size = 51
    v_min = 0
    v_max = 200

    online_net = C51(state_dim, action_dim, atom_size, v_min, v_max)
    target_net = C51(state_dim, action_dim, atom_size, v_min, v_max)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.AdamW(online_net.parameters(), lr=1e-3)

    replay_buffer = []
    capacity = 10000
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    state, _ = env.reset()

    for step in range(50000):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
            with torch.no_grad():
                q = online_net.q_values(state_tensor)
                action = q.argmax(1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Ensure state and next_state are np.ndarray
        state_arr = np.array(state, dtype=np.float32)
        next_state_arr = np.array(next_state, dtype=np.float32)
        replay_buffer.append((state_arr, action, reward, next_state_arr, done))
        if len(replay_buffer) > capacity:
            replay_buffer.pop(0)
        state = next_state if not done else env.reset()[0]
        
        if done:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if len(replay_buffer) < batch_size:
            continue

        samples = np.random.choice(len(replay_buffer), batch_size)
        batch = [replay_buffer[i] for i in samples]
        state_batch = torch.FloatTensor(np.array([b[0] for b in batch]))
        action_indices = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
        reward_batch = torch.FloatTensor([b[2] for b in batch])
        next_state_batch = torch.FloatTensor(np.array([b[3] for b in batch]))
        done_batch = torch.FloatTensor([b[4] for b in batch])

        with torch.no_grad():
            next_prob = target_net(next_state_batch)
            next_q = torch.sum(next_prob * target_net.support, dim=2)
            next_action = next_q.argmax(1)
            next_prob = next_prob[range(batch_size), next_action]

            Tz = reward_batch.unsqueeze(1) + gamma * target_net.support.unsqueeze(0) * (1 - done_batch.unsqueeze(1)) # Tz = r + γz'
            Tz = Tz.clamp(v_min, v_max)
            b = (Tz - v_min) / online_net.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (batch_size - 1) * atom_size, batch_size).unsqueeze(1).expand(batch_size, atom_size).long()
            m = torch.zeros(batch_size, atom_size)
            for i in range(atom_size):
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_prob * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_prob * (b - l.float())).view(-1))

        prob = online_net(state_batch)
        prob_a = prob.gather(1, action_indices.unsqueeze(2).expand(-1, 1, atom_size)).squeeze(1)
        log_prob = torch.log(prob_a + 1e-8)
        loss = -torch.sum(m * log_prob, dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            target_net.load_state_dict(online_net.state_dict())

        if step % 1000 == 0:
            print(f"Step: {step}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.4f}")
