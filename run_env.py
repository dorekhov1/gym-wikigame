import gym

import graph_tool as gt
from agents.simple_agents import *
from agents.a_star_agent import AStarAgent


if __name__ == "__main__":

    env = gym.make("gym_wikigame:wikigame-v0")
    env.seed(2)

    wiki_graph = gt.load_graph("data/processed/graph_with_embeddings.gt")
    env.set_graph(wiki_graph)

    agent = SimpleRandomAgent(wiki_graph)

    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        turns_taken = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            turns_taken += 1
            if done:
                print(f"Turns taken to complete the game: {turns_taken}")
                break
