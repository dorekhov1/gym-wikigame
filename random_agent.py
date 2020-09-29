import random

import gym
import numpy as np


class RandomAgent(object):
    def __init__(self, dim=512):
        self.dim = dim

    def act(self, observation, reward, done):
        return np.random.rand(self.dim)


class SimpleAgent(object):
    def __init__(self, dim=512):
        self.dim = dim

    def act(self, observation, reward, done):
        return observation[1]['title_embedding']


class SimpleRandomAgent(object):
    def __init__(self, dim=512, prob=0.2):
        self.dim = dim
        self.prob = prob

    def act(self, observation, reward, done):
        if random.random() < self.prob:
            return np.random.rand(self.dim)
        return observation[1]['title_embedding']


if __name__ == "__main__":

    env = gym.make("gym_wikigame:wikigame-v0")
    env.seed(0)

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
                print(f'Turns taken to complete the game: {turns_taken}')
                break
