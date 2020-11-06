import random

import numpy as np


class RandomAgent(object):
    def __init__(self, wiki_graph, dim=512):
        self.dim = dim
        self.wiki_graph = wiki_graph
        self.prop_embeddings = self.wiki_graph.vertex_properties["embedding"]

    def act(self, observation, reward, done):
        return {"type": "move", "direction": np.random.rand(self.dim)}


class SimpleAgent(object):
    def __init__(self, wiki_graph, dim=512):
        self.dim = dim
        self.wiki_graph = wiki_graph
        self.prop_embeddings = self.wiki_graph.vertex_properties["embedding"]

    def act(self, observation, reward, done):
        return {"type": "move", "direction": self.prop_embeddings[observation[1]]}


class SimpleRandomAgent(object):
    def __init__(self, wiki_graph, dim=512, prob=0.2):
        self.dim = dim
        self.prob = prob
        self.wiki_graph = wiki_graph
        self.prop_embeddings = self.wiki_graph.vertex_properties["embedding"]

    def act(self, observation, reward, done):
        if random.random() < self.prob:
            return {"type": "move", "direction": np.random.rand(self.dim)}
        return {"type": "move", "direction": self.prop_embeddings[observation[1]]}
