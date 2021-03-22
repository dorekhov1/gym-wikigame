import sys
import random

import gym
import numpy as np
import graph_tool as gt

from reward import *
from config_parser import Configs


class WikigameEnv(gym.Env):

    def __init__(self):
        self.num_pages = None
        self.v_goal = None
        self.v_start = None
        self.v_curr = None
        self.v_curr_neighbors = []
        self.observation_tensor = None

        cf = Configs()
        self.n_steps_away = cf.n_steps_away
        self.random_goal = cf.random_goal

        self.wiki_graph = gt.load_graph(cf.graph_path)

        self.num_pages = self.wiki_graph.num_vertices()
        self.p_titles = self.wiki_graph.vertex_properties["title"]
        self.p_ids = self.wiki_graph.vertex_properties["id"]

        self.reward_fn = getattr(sys.modules[__name__], cf.reward_fn)(self.wiki_graph)
        self.t = 0
        self.max_timesteps = cf.max_timesteps

    def get_random_vertex(self):
        random_index = random.randrange(self.num_pages)
        random_vertex = self.wiki_graph.vertex(random_index)
        return random_vertex

    def get_random_vertex_n_away_from_goal(self, v_goal):
        v = v_goal
        for _ in range(self.n_steps_away):
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
        self.t += 1
        v_new = self.v_curr_neighbors[action]

        done = False
        if v_new == self.v_goal or self.t == self.max_timesteps:
            done = True

        reward = self.reward_fn(v_new, self.v_goal, done)

        self.v_curr = v_new
        self.observation_tensor = self.get_observation_tensor()

        return (
            self.observation_tensor,
            reward,
            done,
            None,
        )

    def reset(self):
        self.t = 0
        self.v_goal = self.get_random_vertex()
        if self.random_goal:
            self.v_start = self.get_random_vertex()
        else:
            self.v_start = self.get_random_vertex_n_away_from_goal(self.v_goal)

        self.reward_fn.reset(self.v_start, self.v_goal)
        self.v_curr = self.v_start

        self.observation_tensor = self.get_observation_tensor()

        return self.observation_tensor

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]
