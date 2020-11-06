from data_prep.data_processing import DataExtractor
from data_prep.embeddings_computation import EmbeddingComputer
from data_prep.graph_creation import GraphCreator


def run_data_extractor():
    data_extractor = DataExtractor()
    print("Processing pages")
    data_extractor.process_pages()
    print("Processing redirects")
    data_extractor.process_redirects()
    print("Processing links")
    data_extractor.process_links()
    print("Removing redirect pages")
    data_extractor.remove_redirect_pages()
    print("Changing keys")
    data_extractor.change_keys_to_title()
    print("Removing invalid links")
    data_extractor.remove_invalid_links()
    print("Removing pages with few links")
    data_extractor.remove_pages_with_few_links()
    print("Saving pages")
    data_extractor.save_pages()


def run_embedding_computer():
    embeddings_generator = EmbeddingComputer()
    print("Loading pages")
    embeddings_generator.load_pages()
    print("Embedding titles")
    embeddings_generator.embed_titles()
    print("Saving pages")
    embeddings_generator.save_data()


def run_graph_creation():
    gc = GraphCreator()
    print("Loading pages")
    gc.load_pages()
    print("Creating vertices")
    gc.create_vertices()
    print("Creating edges")
    gc.create_edges()
    print("Saving graph")
    gc.save_graph()


if __name__ == "__main__":
    print("Running data extractor\n")
    run_data_extractor()

    print("\nRunning embedding computer\n")
    run_embedding_computer()

    print("\nRunning graph creation\n")
    run_graph_creation()
