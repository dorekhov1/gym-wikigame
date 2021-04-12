import numpy as np

from abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, wiki_graph=None, kwargs=None):
        super().__init__(wiki_graph, kwargs)

    def act(self, state):
        links = state[1:]

        return np.argwhere(links == np.random.choice(state[1:]))[0][0]

    def update(self, reward, done):
        pass

    def save(self, solved=False):
        pass
