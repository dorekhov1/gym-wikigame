import pickle

from tqdm import tqdm
from graph_tool.centrality import *
from graph_tool.all import *
from graph_tool.topology import *


class GraphCreator:
    def __init__(self):
        self.pages = {}
        self.page_title_to_vertex_map = {}
        self.g = Graph()
        v_title = self.g.new_vertex_property("string")
        v_id = self.g.new_vertex_property("int")
        self.g.vp.title = v_title
        self.g.vp.id = v_id

    def load_pages(self):
        with open("data/processed/pages.pickle", "rb") as handle:
            self.pages = pickle.load(handle)

    def create_vertices(self):
        missed_pages = 0
        for page_title in tqdm(self.pages.keys()):
            try:
                page = self.pages[page_title]
                page_vertex = self.g.add_vertex()
                self.g.vp.title[page_vertex] = page["proper_title"]
                self.g.vp.id[page_vertex] = page["id"]
                self.page_title_to_vertex_map[page_title] = page_vertex
            except:
                missed_pages += 1
        print(f"Missed pages: {missed_pages}")

    def create_edges(self):
        missed_edges = 0
        for page_title in tqdm(self.pages.keys()):
            try:
                page = self.pages[page_title]
                page_vertex = self.page_title_to_vertex_map[page_title]
                links = page["links"]

                for link in links:
                    link_page = self.page_title_to_vertex_map[link]
                    self.g.add_edge(page_vertex, link_page)

                del page["links"]
            except:
                missed_edges += 1
        print(f"Missed edges: {missed_edges}")

    def get_highest_centrality(self, n=200, centrality_func=pagerank):
        centrality_values = centrality_func(self.g).get_array()
        nth_biggest_value = centrality_values[centrality_values.argsort()[-n]]
        filter_values = centrality_values >= nth_biggest_value
        filter_property = self.g.new_vertex_property('bool', filter_values)
        self.g.set_vertex_filter(filter_property)
        self.g.purge_vertices()
        self.g.clear_filters()
        self.g = Graph(self.g, prune=True)

    def extract_largest_component(self):
        self.g = extract_largest_component(self.g, prune=True, directed=True)
        self.g = Graph(self.g, prune=True)

    def list_titles(self):
        for v in self.g.vertices():
            print(self.g.vp.title[v])

    def save_graph(self):
        self.g.save("data/processed/wiki_graph.gt")


if __name__ == "__main__":
    gc = GraphCreator()

    print("loading pages")
    gc.load_pages()

    print("creating vertices")
    gc.create_vertices()

    print("creating edges")
    gc.create_edges()

    print('pruning based on centrality')
    gc.get_highest_centrality()

    print("extracting the largest component")
    gc.extract_largest_component()

    print("saving graph")
    gc.save_graph()
