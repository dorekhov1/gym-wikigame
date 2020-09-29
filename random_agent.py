import gym

import numpy as np


class RandomAgent(object):
    def __init__(self, dim=512):
        self.dim = dim

    def act(self, observation, reward, done):
        return np.random.rand(self.dim,)


if __name__ == "__main__":

    env = gym.make("gym_wikigame:wikigame-v0")
    env.seed(0)

    agent = RandomAgent()

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
