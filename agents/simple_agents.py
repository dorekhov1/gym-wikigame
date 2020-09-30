import random

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
        return observation[1]["title_embedding"]


class SimpleRandomAgent(object):
    def __init__(self, dim=512, prob=0.2):
        self.dim = dim
        self.prob = prob

    def act(self, observation, reward, done):
        if random.random() < self.prob:
            return np.random.rand(self.dim)
        return observation[1]["title_embedding"]
