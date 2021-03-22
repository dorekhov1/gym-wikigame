from graph_tool.topology import shortest_distance


class SimpleRewardFn:
    def __init__(self, graph):
        pass

    def reset(self, v_start, v_goal):
        pass

    def __call__(self, v_new, v_goal, done):
        return int(done)


class PunishingRewardFn:

    def __init__(self, graph):
        self.graph = graph
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
