import pickle
import random


class DataHandler:
    def __init__(self):
        with open("data/wiki_with_embeddings.pickle", "rb") as handle:
            self.data = pickle.load(handle)

    def get_refs_with_embeddings(self, title):
        curr_refs = self.data[title]["refs"]
        curr_refs_embeddings = [
            self.data[ref]["title_embedding"] for ref in curr_refs
        ]
        return curr_refs, curr_refs_embeddings

    def get(self, title):
        return self.data[title]

    def get_random_page(self):
        return random.choice(list(self.data))