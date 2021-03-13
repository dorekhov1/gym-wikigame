import pickle

import torch
from torch import nn
import graph_tool as gt
from tqdm import tqdm


class TorchEmbeddingBuilder:

    def __init__(self):
        self.wiki_graph = None
        self.embedding = None

    def load_graph(self):
        self.wiki_graph = gt.load_graph("data/processed/graph_with_embeddings.gt")

    def construct_embeddings(self):
        embeddings_list = []
        vp_embedding = self.wiki_graph.vertex_properties["embedding"]
        for v in tqdm(self.wiki_graph.vertices()):
            title_embedding = vp_embedding[v]
            embeddings_list.append(title_embedding)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings_list), freeze=False)
        del self.wiki_graph.vertex_properties["embedding"]

    def save_results(self):
        with open("data/processed/torch_embeddings.pickle", "wb+") as handle:
            pickle.dump(self.embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.wiki_graph.save("data/processed/wiki_graph.gt")


if __name__ == "__main__":
    builder = TorchEmbeddingBuilder()
    builder.load_graph()
    builder.construct_embeddings()
    builder.save_results()
