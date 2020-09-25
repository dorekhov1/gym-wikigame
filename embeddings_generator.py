import pickle
import time

import tqdm
import tensorflow as tf
import tensorflow_hub as hub


class EmbeddingsGenerator:
    def __init__(self):
        self.batch_size = 512
        self.wiki_data_file = "data/wiki.pickle"
        self.wiki_data_file_with_embeddings = "data/wiki_with_embeddings.pickle"

        tf.config.set_visible_devices([], 'GPU')

        print("Loading data")
        start_time = time.time()
        with open(self.wiki_data_file, "rb") as handle:
            self.wiki_data = pickle.load(handle)
        print(f"Time to load data: {time.time() - start_time}")

        print("Loading model")
        start_time = time.time()
        self.embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        print(f"Time to load model: {time.time() - start_time}")

    def embed_titles(self):
        batch_page_titles = []
        for page_title in tqdm.tqdm(self.wiki_data):
            batch_page_titles.append(page_title)
            if len(batch_page_titles) == self.batch_size:
                batch_title_embeddings = self.embed(batch_page_titles)
                for i, processed_page_title in enumerate(batch_page_titles):
                    page = self.wiki_data[processed_page_title]
                    page["title_embedding"] = batch_title_embeddings[i].numpy()
                batch_page_titles = []

    def save_data(self):
        with open(self.wiki_data_file_with_embeddings, "wb+") as handle:
            pickle.dump(self.wiki_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    embeddings_generator = EmbeddingsGenerator()
    embeddings_generator.embed_titles()
    embeddings_generator.save_data()
