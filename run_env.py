import gym
from gym_wikigame.envs.data_handler import DataHandler

from agents.simple_agents import *


if __name__ == "__main__":

    env = gym.make("gym_wikigame:wikigame-v0")
    env.seed(0)

    data_handler = DataHandler()
    env.set_data_handler(data_handler)

    agent = SimpleRandomAgent()

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
