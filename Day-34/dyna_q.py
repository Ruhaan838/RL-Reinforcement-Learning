import numpy as np
import random

class GridWorld:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            y = min(self.height - 1, y + 1)
        elif action == 1:  # down
            y = max(0, y - 1)
        elif action == 2:  # right
            x = min(self.width - 1, x + 1)
        elif action == 3:  # left
            x = max(0, x - 1)

        self.state = (x, y)
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return self.state, reward, done

class DynaQAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        self.actions = [0, 1, 2, 3]  # up, down, right, left
        self.Q = {}
        self.model = {}

    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        qs = [self.get_Q(state, a) for a in self.actions]
        return self.actions[np.argmax(qs)]

    def update(self, state, action, reward, next_state):
        max_q_next = max([self.get_Q(next_state, a) for a in self.actions])
        old_q = self.get_Q(state, action)
        self.Q[(state, action)] = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)

        self.model[(state, action)] = (reward, next_state)

        # planning
        for _ in range(self.planning_steps):
            s, a = random.choice(list(self.model.keys()))
            r, s_next = self.model[(s, a)]
            max_q_next = max([self.get_Q(s_next, a2) for a2 in self.actions])
            old_q = self.get_Q(s, a)
            self.Q[(s, a)] = old_q + self.alpha * (r + self.gamma * max_q_next - old_q)

env = GridWorld(width=5, height=5, start=(0, 0), goal=(4, 4))
agent = DynaQAgent(env)

episodes = 50

for ep in range(episodes):
    state = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        steps += 1
    print(f"Episode {ep + 1}: reached goal in {steps} steps")
