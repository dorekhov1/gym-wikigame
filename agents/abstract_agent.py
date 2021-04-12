from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    @abstractmethod
    def __init__(self, wiki_graph, kwargs):
        self.wiki_graph = wiki_graph

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update(self, reward, done):
        pass

    @abstractmethod
    def save(self, solved):
        pass
