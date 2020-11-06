import random

import gym
import numpy as np
from graph_tool.topology import shortest_path, shortest_distance

class WikigameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.wiki_graph = None
        self.num_pages = None
        self.prop_titles = None
        self.prop_ids = None
        self.prop_embeddings = None
        self.v_goal_state = None
        self.v_start_state = None
        self.v_curr_state = None
        self.random_goal = False
        self.n = 10

    def set_graph(self, wiki_graph):
        self.wiki_graph = wiki_graph
        self.num_pages = self.wiki_graph.num_vertices()
        self.prop_titles = self.wiki_graph.vertex_properties["title"]
        self.prop_ids = self.wiki_graph.vertex_properties["id"]
        self.prop_embeddings = self.wiki_graph.vertex_properties["embedding"]

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
        embeddings = list(map(lambda v: self.prop_embeddings[v], v_links))
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
        if action["type"] == "return":
            self.v_curr_state = action["returning_state"]
            print(f"Returning: {action['returning_state']}")
            done, reward = False, 0
        else:
            (v_links, embeddings,) = self.get_links_and_embeddings(self.v_curr_state)
            dot_products = [
                self.compute_dot_product(action["direction"], emb) for emb in embeddings
            ]
            v_closest_page = v_links[np.argmax(dot_products)]

            print(f"Navigating: {self.prop_titles[v_closest_page]}")

            done = False
            if v_closest_page == self.v_goal_state:
                done = True
            reward = int(done)
            self.v_curr_state = v_closest_page

        return (
            (self.v_curr_state, self.v_goal_state,),
            reward,
            done,
            None,
        )

    def reset(self):
        self.v_start_state = self.get_random_vertex()
        if self.random_goal:
            self.v_goal_state = self.get_random_vertex()
        else:
            self.v_goal_state = self.get_random_vertex_n_away_from_source(self.v_start_state)

        print(f"Start state: {self.prop_titles[self.v_start_state]}")
        print(f"Goal state: {self.prop_titles[self.v_goal_state]}")

        shortest_dist = shortest_distance(self.wiki_graph, self.v_start_state, self.v_goal_state, directed=True)
        print(f"Shortest distance is {shortest_dist}")

        self.v_curr_state = self.v_start_state
        return self.v_curr_state, self.v_goal_state

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]
