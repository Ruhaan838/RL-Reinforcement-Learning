from agent import Agent
import gym

if __name__ == "__main__":
    env = gym.make("PongNoFrameskip-v4")
    agent = Agent(env, double=True, dueling=True)
    agent.train(500)