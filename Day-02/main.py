from grid_world import GridWorld
from MDP import MDPModel
from agent import ValueIterationAgent

world = GridWorld(rows=3, cols=3, terminal_state=(2, 2))
mdp = MDPModel(world)
agent = ValueIterationAgent(mdp)

agent.run()

print("State Values:")
print(agent.get_value_grid())

print("\nOptimal Policy (0=Up, 1=Down, 2=Left, 3=Right):")
print(agent.get_policy_grid())
