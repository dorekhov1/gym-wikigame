import gym
import graph_tool as gt

from config_parser import Configs
from agents.random_agent import *


def main():

    cf = Configs()

    env = gym.make(cf.env_name)
    env.seed(cf.random_seed)

    wiki_graph = gt.load_graph(cf.graph_path)
    env.set_graph(wiki_graph)

    agent = SimpleAgent(wiki_graph)

    done = False

    for i in range(cf.max_episodes):
        observation = env.reset()
        for timesteps_taken in range(cf.max_timesteps):
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            if done:
                print(f"Turns taken to complete the game: {timesteps_taken+1}")
                break


if __name__ == "__main__":
    main()
