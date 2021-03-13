import random

import gym
import numpy as np
import graph_tool as gt
from graph_tool.topology import shortest_distance

from config_parser import Configs


class WikigameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.num_pages = None
        self.v_goal = None
        self.v_start = None
        self.v_curr = None
        self.v_curr_neighbors = []
        self.observation_tensor = None
        self.random_goal = False
        self.num_links = 1
        self.n = 1

        self.shortest_dist = None

        cf = Configs('PPO')

        self.wiki_graph = gt.load_graph(cf.graph_path)

        self.num_pages = self.wiki_graph.num_vertices()
        self.p_titles = self.wiki_graph.vertex_properties["title"]
        self.p_ids = self.wiki_graph.vertex_properties["id"]

    def get_random_vertex(self):
        random_index = random.randrange(self.num_pages)
        random_vertex = self.wiki_graph.vertex(random_index)
        return random_vertex

    def get_random_vertex_n_away_from_goal(self, v_goal):
        v = v_goal
        for _ in range(self.n):
            v_links = list(v.in_neighbours())
            v = random.choice(v_links)
        return v

    def get_observation_tensor(self):
        observation_tensor = [int(self.v_goal)]

        self.v_curr_neighbors = []
        for v in self.v_curr.out_neighbors():
            self.v_curr_neighbors.append(v)
            observation_tensor.append(int(v))

        return np.array(observation_tensor)

    def step(self, action: int):
        # print(f'action: {action}, num_neighbours: {len(self.v_curr_neighbors)}')
        v_new = self.v_curr_neighbors[action]

        done = False
        if v_new == self.v_goal:
            done = True

        new_shortest_dist = shortest_distance(
            self.wiki_graph, v_new, self.v_goal, directed=True
        )

        if new_shortest_dist > self.shortest_dist:
            reward = -2
        elif new_shortest_dist == self.shortest_dist:
            reward = -1
        else:
            reward = int(done)

        # reward = int(done)

        self.shortest_dist = new_shortest_dist

        self.v_curr = v_new
        self.num_links = len(list(self.v_curr.out_edges()))
        self.observation_tensor = self.get_observation_tensor()

        return (
            self.observation_tensor,
            reward,
            done,
            None,
        )

    def reset(self):
        self.v_goal = self.get_random_vertex()
        if self.random_goal:
            self.v_start = self.get_random_vertex()
        else:
            self.v_start = self.get_random_vertex_n_away_from_goal(self.v_goal)

        assert self.v_goal in list(self.v_start.out_neighbours())

        # print(f"Start: {self.p_titles[self.v_start]}")
        # print(f"Goal: {self.p_titles[self.v_goal]}")

        self.shortest_dist = shortest_distance(
            self.wiki_graph, self.v_start, self.v_goal, directed=True
        )
        # print(f"Shortest distance is {self.shortest_dist}")

        self.v_curr = self.v_start
        self.num_links = len(list(self.v_curr.out_edges()))

        self.observation_tensor = self.get_observation_tensor()

        return self.observation_tensor

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]
