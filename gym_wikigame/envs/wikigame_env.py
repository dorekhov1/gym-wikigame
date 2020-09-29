import random
import pickle

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class WikigameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.goal_state = None
        self.start_state = None
        self.current_state = None

        with open("data/wiki_with_embeddings.pickle", "rb") as handle:
            self.data = pickle.load(handle)

    def find_title_embedding(self, title):
        try:
            return self.data[title]["title_embedding"]
        except KeyError:
            # TODO handle this error properly (need a list of aliases for each page)
            print(f"Key not found: {title}")
            return None

    @staticmethod
    def compute_dot_product(action, reference_emb):
        if reference_emb is None:
            return 0
        return np.dot(action, reference_emb)

    def step(self, action):
        current_references = self.data[self.current_state]["refs"]
        current_references_embeddings = [self.find_title_embedding(ref) for ref in current_references]
        dot_products = [self.compute_dot_product(action, emb) for emb in current_references_embeddings]
        closest_page = current_references[np.argmax(dot_products)]

        done = False
        if closest_page == self.goal_state:
            done = True
        reward = int(done)
        self.current_state = closest_page

        return closest_page, reward, done, None

    def reset(self):
        self.start_state = random.choice(list(self.data))
        self.goal_state = random.choice(list(self.data))

        self.current_state = self.start_state

        return (
            self.data[self.current_state]["title_embedding"],
            self.data[self.goal_state]["title_embedding"],
        )

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]


if __name__ == "__main__":
    e = WikigameEnv()
    print("done")
