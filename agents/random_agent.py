import numpy as np


class RandomAgent(object):
    def __init__(self, wiki_graph, kwargs):
        pass

    def act(self, state):
        links = state[1:]

        return np.argwhere(links == np.random.choice(state[1:]))[0][0]

    def update(self, reward, done):
        pass

    def save(self, solved=False):
        pass
