import pickle

from tqdm import tqdm
from graph_tool.all import *


class GraphCreator:
    def __init__(self):
        self.pages = {}
        self.page_title_to_vertex_map = {}
        self.g = Graph()
        v_title = self.g.new_vertex_property("string")
        v_id = self.g.new_vertex_property("int")
        v_embeddings = self.g.new_vertex_property("object")
        self.g.vp.title = v_title
        self.g.vp.id = v_id
        self.g.vp.embedding = v_embeddings

    def load_pages(self):
        with open("data/processed/page_embeddings.pickle", "rb") as handle:
            self.pages = pickle.load(handle)

    def create_vertices(self):
        for page_title in tqdm(self.pages.keys()):
            page = self.pages[page_title]
            page_vertex = self.g.add_vertex()
            self.g.vp.title[page_vertex] = page_title
            self.g.vp.id[page_vertex] = page['id']
            self.g.vp.embedding[page_vertex] = page['title_embedding']
            self.page_title_to_vertex_map[page_title] = page_vertex

            del page['title_embedding']
            del page['id']

    def create_edges(self):
        for page_title in tqdm(self.pages.keys()):
            page = self.pages[page_title]
            page_vertex = self.page_title_to_vertex_map[page_title]
            links = page['links']

            for link in links:
                link_page = self.page_title_to_vertex_map[link]
                self.g.add_edge(page_vertex, link_page)

            del page['links']

    def save_graph(self):
        self.g.save("data/processed/graph_with_embeddings.gt")


if __name__ == "__main__":
    gc = GraphCreator()

    print("loading pages")
    gc.load_pages()

    print("creating vertices")
    gc.create_vertices()

    print("creating edges")
    gc.create_edges()

    print("saving graph")
    gc.save_graph()
