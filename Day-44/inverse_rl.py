import gymnasium as gym
import numpy as np
from scipy.special import logsumexp

def generate_expert(env, policy, n_trajectories=20, max_steps=20):
    demos = []
    for _ in range(n_trajectories):
        s, _ = env.reset()
        traj = []
        for _ in range(max_steps):
            a = policy[s]
            traj.append((s, a))
            s, _, done, _, _ = env.step(a)
            if done:
                break
        demos.append(traj)
    return demos

def feature_matrix(n_states):
    return np.eye(n_states)

def soft_value_iteration(P, R, gamma=0.9, eps=1e-5):
    V = np.zeros(len(R))
    while True:
        Q = np.zeros((len(R), P.shape[1]))
        for a in range(P.shape[1]):
            Q[:, a] = R + gamma * P[:, a, :] @ V
        V_new = logsumexp(Q, axis=1)
        if np.max(np.abs(V_new - V)) < eps:
            break
        V = V_new
    pi = np.exp(Q - V[:, None])
    return pi

def expected_svf(P, pi, start_dist, horizon=20):
    mu = start_dist.copy()
    svf = np.zeros_like(start_dist)
    for _ in range(horizon):
        svf += mu
        mu_new = np.zeros_like(mu)
        for s in range(len(mu)):
            for a in range(P.shape[1]):
                mu_new += mu[s] * pi[s, a] * P[s, a]
        mu = mu_new
    return svf

env = gym.make("FrozenLake-v1", is_slippery=False)
env = env.unwrapped
nS = env.observation_space.n
nA = env.action_space.n

P = np.zeros((nS, nA, nS))
for s in range(nS):
    for a in range(nA):
        for prob, ns, _, _ in env.P[s][a]:
            P[s, a, ns] += prob

expert_policy = np.zeros(nS, dtype=int)
for s in range(nS):
    expert_policy[s] = 2 if s % 4 != 3 else 1 

demos = generate_expert(env, expert_policy)
F = feature_matrix(nS)

feat_exp_expert = np.zeros(nS)
for traj in demos:
    for s, _ in traj:
        feat_exp_expert += F[s]
feat_exp_expert /= len(demos)

w = np.zeros(nS)
start_dist = np.zeros(nS)
start_dist[0] = 1.0
lr = 0.1
for _ in range(50):
    R = F @ w
    pi = soft_value_iteration(P, R)
    svf = expected_svf(P, pi, start_dist)
    grad = feat_exp_expert - F.T @ svf / len(demos)
    w += lr * grad

print("Learned reward:", np.round(w, 3).reshape(4, 4))
