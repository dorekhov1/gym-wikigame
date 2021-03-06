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

    def delete_pages(self, pages_to_remove):
        for page_to_remove in pages_to_remove:
            del self.pages[page_to_remove]

        for page_title in self.pages:
            for page_to_remove in pages_to_remove:
                if page_to_remove in self.pages[page_title]['links']:
                    self.pages[page_title]['links'].remove(page_to_remove)

    def embed_titles(self):
        batch_page_titles = []
        clean_batch_page_titles = []
        missed_titles = 0
        titles = self.pages.keys()
        titles_to_delete = []
        for page_title in tqdm(titles):

            try:
                clean_title = self.pages[page_title]['proper_title']
            except:
                missed_titles += 1
                titles_to_delete.append(page_title)
                continue

            batch_page_titles.append(page_title)
            clean_batch_page_titles.append(clean_title)
            if len(batch_page_titles) == self.batch_size:
                self.embed_batch(batch_page_titles, clean_batch_page_titles)
                batch_page_titles = []
        print(f'Missed titles: {missed_titles}')
        self.embed_batch(batch_page_titles, clean_batch_page_titles)
        self.delete_pages(titles_to_delete)

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
