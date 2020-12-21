import pickle
import time

from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub


class EmbeddingComputer:
    def __init__(self):
        self.batch_size = 512
        self.pages = {}

        self.embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        tf.config.set_visible_devices([], 'GPU')

    def load_pages(self):
        with open("data/processed/pages.pickle", "rb") as handle:
            self.pages = pickle.load(handle)

    def embed_titles(self):
        batch_page_titles = []
        clean_batch_page_titles = []
        for page_title in tqdm(self.pages):

            clean_title = page_title.replace("_", " ").replace("\\", "")

            batch_page_titles.append(page_title)
            clean_batch_page_titles.append(clean_title)
            if len(batch_page_titles) == self.batch_size:
                self.embed_batch(batch_page_titles, clean_batch_page_titles)
                batch_page_titles = []
        self.embed_batch(batch_page_titles, clean_batch_page_titles)

    def embed_batch(self, batch_page_titles, clean_batch_page_titles):
        batch_title_embeddings = self.embed(clean_batch_page_titles)
        for i, processed_page_title in enumerate(batch_page_titles):
            page = self.pages[processed_page_title]
            page["title_embedding"] = batch_title_embeddings[i].numpy()

    def save_data(self):
        with open("data/processed/page_embeddings.pickle", "wb+") as handle:
            pickle.dump(self.pages, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    embeddings_generator = EmbeddingComputer()
    embeddings_generator.load_pages()
    embeddings_generator.embed_titles()
    embeddings_generator.save_data()
