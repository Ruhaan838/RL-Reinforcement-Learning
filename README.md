# RL Fundamentals
## Day - 01

- Learns what is RL? and learn about Agenet, Env, Rewards, Policy.
- code: GridWorld simulation.

## Day - 02

- Learn about
  - Markov Process
  - Markov Reward Processes
  - Markov Decision Processes
- Code: Gridworld MDP simulation.

## Day - 03
- Learn about
  - Policy Evaluation using temproal difference <br>
  $V(s) \leftarrow V(s) + \alpha (r_t + \gamma V(s') - V(s))$ <br>
  $V(s)$ : State Value Funtion <br>
  $\alpha$ : Constant <br>
  $r_t$ : reward <br>
  $s'$ : new state <br>
  $\gamma$ :  discount factor
- Code: Policy Evaluation using temproal difference

## Day - 04
- Bellman Equation and its optimal policy function
- Code: Get optimal Value and policy using Bellman eqation

## Day - 05
- Policy Iteration
  - two stage process in a markov Decision Process (1. Policy Evaluation and 2. Policy Improvement)

- Code: Policy Iteration in grid world.

## Day - 06
- Value Iteration

- Code: Value Iteration in grid world.

## Day - 07

- Epsilon Greedy
  - eps greedy is very intersting policy algorithm.
- Code: Implement Epsilon Greedy Algorithm for Treasure Trap Envirement.


## Day - 08

- MCP(Monte Carlo Prediction)

- Code: simple implementation of MCP 

## Day - 09

- MC Control for Ad-click simulation

- Code: MC control for Ad-click simulation

## Day - 10

- Temporal-Difference (TD) Learning - TD(0)
- Code: using gym envirement "FrozenLake-v1" implement the TD-0

## Day - 11

- SARSA is an on-policy TD(temporal-Diffrence) control algorithm.
- Code: using gym envirement "FrozenLake-v1" implement the SARASA.

## Day - 12
- Learn Q Learning
- Code: using gym Taxi-v3 envirement implement the Q learning elgorithm.

## Day - 13
- On-policy vs Off-policy
- Code: Compare SARSA and Q-Learning	Conceptual and coding differences

# Deep RL

## Day - 14
- REINFORCE Algorithm
- Code REINFORCE algo in pytorch.

## Day - 15
- REINFORCE with baseline 
- Code REINFORCE with baseline algo in pytorch using gym envirement CartPole-v1

## Day - 16
- Actor-Critic methods.
- Code: Implement Basic Actor-Critic.

## Day - 17

- Deep Q-Learning
- Code Deep Q-Learning using pytorch for CartPole envirement.

## Day - 18

- Dueling DQN


- Standard DQN:
  - Input -> Dense Layers -> Output (num_actions)
- Dueling DQN:
  - Input -> Shared Dense Layers -> Value Stream -> V(s) -> Advantage Stream -> A(s,a) <br>
    Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))

## Day - 19

- Double DQN:
- Implement the Double DQN using MountainCar-v0 envirement from gym.

## Day - 20

- Prioritized Replay
- Implement PER using pytorch and gym

## Day - 21
- Prectice on other envirement.
- Implement the DQN on LunarLander using torch and gym.

## Day - 22
- Implement the DQN, Dueling DQN, Double DQN in PongNoFrameskip from gym.

## Day - 23
- Implement the A2C(Advantage Actor-Critic) in Pytorch using gym env.

## Day - 24
- Implement the A3C(Asynchronous Advantage Actor-Critic) in Pytorch using gym env.

## Day - 25
- Implement the trpo in pytorch.

## Day - 26
- Implement PPO in pendulum env (failed).

## Day - 27
- Implement PPO in pendulum gym env.

## Day - 28
- Implement the PPO and TRPO in Hopper-v5 env.

## Day - 29
- Implement the CNN + PPO for CarRacing-v3

# Advanced RL Techniques.

## Day - 30
- Implement the C51 Algorithm.

## Day - 31
- Noisy Networks to a DQN using torch.

## Day - 32
- Implement the rainbow DQN in torch.