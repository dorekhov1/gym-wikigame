import random

import gym
import gym.spaces
import numpy as np
import graph_tool as gt
from graph_tool.topology import shortest_distance

from config_parser import Configs


class WikigameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.wiki_graph = None
        self.num_pages = None
        self.p_titles = None
        self.p_ids = None
        self.p_embeddings = None
        self.v_goal = None
        self.v_start = None
        self.v_curr = None
        self.random_goal = False
        self.n = 1

        self.shortest_dist = None
        self.observation_space = gym.spaces.Box(0, 1, shape=(1024,))
        self.action_space = gym.spaces.Box(0, 1, shape=(512,))

        cf = Configs('PPO')

        self.wiki_graph = gt.load_graph(cf.graph_path)

        self.num_pages = self.wiki_graph.num_vertices()
        self.p_titles = self.wiki_graph.vertex_properties["title"]
        self.p_ids = self.wiki_graph.vertex_properties["id"]
        self.p_embeddings = self.wiki_graph.vertex_properties["embedding"]

    def get_random_vertex(self):
        idx = random.randrange(self.num_pages + 1)
        v = self.wiki_graph.vertex(idx)
        return v

    def get_random_vertex_n_away_from_source(self, v_source):
        v = v_source
        for _ in range(self.n):
            v_links = list(v.out_neighbors())
            v = random.choice(v_links)
        return v

    def get_links_and_embeddings(self, v_source):
        v_links = list(v_source.out_neighbors())
        embeddings = list(map(lambda v: self.p_embeddings[v], v_links))
        return v_links, embeddings

    @staticmethod
    def compute_dot_product(action, ref_emb):
        try:
            return np.dot(action, ref_emb)
        except:
            # TODO not sure why this is happening, need to investigate
            print("problem?")
            return np.dot(action, np.squeeze(ref_emb))

    def step(self, action: dict):

        (v_links, embeddings,) = self.get_links_and_embeddings(self.v_curr)
        dot_products = [self.compute_dot_product(action, emb) for emb in embeddings]
        v_closest_page = v_links[np.argmax(dot_products)]

        # print(f"Navigating: {self.prop_titles[v_closest_page]}")

        done = False
        if v_closest_page == self.v_goal:
            done = True

        new_shortest_dist = shortest_distance(
            self.wiki_graph, self.v_start, self.v_goal, directed=True
        )

        if new_shortest_dist > self.shortest_dist:
            reward = -2
        elif new_shortest_dist == self.shortest_dist:
            reward = -1
        else:
            reward = int(done)

        # reward = int(done)

        self.shortest_dist = new_shortest_dist

        self.v_curr = v_closest_page

        return (
            (self.v_curr, self.v_goal,),
            reward,
            done,
            None,
        )

    def reset(self):
        self.v_start = self.get_random_vertex()
        if self.random_goal:
            self.v_goal = self.get_random_vertex()
        else:
            self.v_goal = self.get_random_vertex_n_away_from_source(self.v_start)

        num_links = len(list(self.v_start.out_edges()))
        # print(f"Start: {self.p_titles[self.v_start]}, #links: {num_links}")
        # print(f"Goal: {self.p_titles[self.v_goal]}")

        self.shortest_dist = shortest_distance(
            self.wiki_graph, self.v_start, self.v_goal, directed=True
        )
        # print(f"Shortest distance is {self.shortest_dist}")

        self.v_curr = self.v_start
        return self.v_curr, self.v_goal

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]
