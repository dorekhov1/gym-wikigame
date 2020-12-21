import random

import numpy as np


class RandomAgent(object):
    def __init__(self, wiki_graph, dim=512):
        self.dim = dim
        self.wiki_graph = wiki_graph
        self.prop_embeddings = self.wiki_graph.vertex_properties["embedding"]

    def act(self, observation):
        return np.random.rand(self.dim)


class SimpleAgent(object):
    def __init__(self, wiki_graph, dim=512):
        self.dim = dim
        self.wiki_graph = wiki_graph
        self.prop_embeddings = self.wiki_graph.vertex_properties["embedding"]

    def act(self, observation):
        return self.prop_embeddings[observation[1]]


class SimpleRandomAgent(object):
    def __init__(self, wiki_graph, dim=512, prob=0.2):
        self.dim = dim
        self.prob = prob
        self.wiki_graph = wiki_graph
        self.prop_embeddings = self.wiki_graph.vertex_properties["embedding"]

    def act(self, observation):
        if random.random() < self.prob:
            return np.random.rand(self.dim)
        return self.prop_embeddings[observation[1]]
