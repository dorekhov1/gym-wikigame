from abc import ABC, abstractmethod

from graph_tool.topology import shortest_distance


class AbstractRewardFn(ABC):
    @abstractmethod
    def __init__(self, graph):
        self.graph = graph

    @abstractmethod
    def reset(self, v_start, v_goal):
        pass

    @abstractmethod
    def __call__(self, v_new, v_goal, done):
        pass


class SimpleRewardFn(AbstractRewardFn):
    def __init__(self, graph):
        super().__init__(graph)

    def reset(self, v_start, v_goal):
        pass

    def __call__(self, v_new, v_goal, done):
        return int(done)


class PunishingRewardFn(AbstractRewardFn):

    def __init__(self, graph):
        super().__init__(graph)
        self.shortest_dist = None

    def reset(self, v_start, v_goal):
        self.shortest_dist = shortest_distance(
            self.graph, v_start, v_goal, directed=True
        )

    def __call__(self, v_new, v_goal, done):
        new_shortest_dist = shortest_distance(
            self.graph, v_new, v_goal, directed=True
        )

        if new_shortest_dist > self.shortest_dist:
            reward = -2
        elif new_shortest_dist == self.shortest_dist:
            reward = -1
        else:
            reward = int(done)

        self.shortest_dist = new_shortest_dist

        return reward
