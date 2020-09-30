import random

import gym
import numpy as np


class WikigameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.goal_state = None
        self.start_state = None
        self.curr_state = None
        self.data_handler = None

    def set_data_handler(self, data_handler):
        self.data_handler = data_handler

    @staticmethod
    def compute_dot_product(action, ref_emb):
        try:
            return np.dot(action, ref_emb)
        except:
            # TODO not sure why this is happening, need to investigate
            return np.dot(action, np.squeeze(ref_emb))

    def step(self, action):
        curr_refs, curr_refs_embeddings = self.data_handler.get_refs_with_embeddings(
            self.curr_state
        )
        dot_products = [
            self.compute_dot_product(action, emb) for emb in curr_refs_embeddings
        ]
        closest_page = curr_refs[np.argmax(dot_products)]

        print(f"Navigating: {closest_page}")

        done = False
        if closest_page == self.goal_state:
            done = True
        reward = int(done)
        self.curr_state = closest_page

        return (
            (
                self.data_handler.get(self.curr_state),
                self.data_handler.get(self.goal_state),
            ),
            reward,
            done,
            None,
        )

    def reset(self):
        self.start_state = self.data_handler.get_random_page()
        self.goal_state = self.data_handler.get_random_page()

        print(f"Start state: {self.start_state}\nGoal state: {self.goal_state}\n")

        self.curr_state = self.start_state

        return (
            self.data_handler.get(self.curr_state),
            self.data_handler.get(self.goal_state),
        )

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]
