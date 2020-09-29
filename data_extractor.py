import os
import re
import json
import pickle
from urllib.parse import unquote
from multiprocessing import Pool

import tqdm as tqdm
import numpy as np


class DataExtractor:
    def __init__(self):
        self.data_dir = "data/extracted"
        self.href_regex = r"<a href=\"(.*?)\".*?<\/a>"
        self.min_number_of_references = 5

        self.all_docs = {}
        self.all_missed_refs = []
        self.pages_with_few_references = []

    def files_generator(self):
        for directory in os.listdir(self.data_dir):
            dir_path = os.path.join(self.data_dir, directory)
            if os.path.isdir(dir_path):
                for file in os.listdir(dir_path):
                    if file.startswith("wiki_"):
                        yield os.path.join(dir_path, file)

    def extract_data(self, file_path):
        data = {}
        with open(file_path) as f:
            for line in f:
                wiki_doc = json.loads(line)
                wiki_doc["refs"] = [
                    unquote(ref)
                    for ref in re.findall(self.href_regex, wiki_doc["text"])
                ]
                data[wiki_doc["title"]] = wiki_doc

                del wiki_doc["text"]
                del wiki_doc["url"]

        return data

    def run_extract_data(self):
        files = list(self.files_generator())
        all_docs = {}
        with Pool(processes=15) as pool:
            with tqdm.tqdm(total=len(files)) as pbar:
                for i, docs in enumerate(pool.imap_unordered(self.extract_data, files)):
                    all_docs.update(docs)
                    pbar.update()
        self.all_docs = all_docs

    def check_references(self, doc):
        missed_references = []

        for ref in doc["refs"]:
            try:
                _ = self.all_docs[ref]
            except KeyError:
                # TODO this is not good - some of the missed references are just page redirects
                #  (e.g. "WW1" redirects to "World War 1") - I will delete all missed references
                #  for now for the sake of simplicity, but eventually it will be beneficial to
                #  figure out how to resolve those misses.
                missed_references.append(ref)
                doc["refs"].remove(ref)

        return missed_references

    def run_check_references(self, add_to_missed_refs):
        all_missed_refs = []
        for doc in tqdm.tqdm(self.all_docs.values()):
            missed_refs = self.check_references(doc)
            if add_to_missed_refs:
                all_missed_refs.extend(missed_refs)
        if add_to_missed_refs:
            self.all_missed_refs = set(all_missed_refs)

    def remove_page_if_few_references(self, title):
        if len(self.all_docs[title]["refs"]) < self.min_number_of_references:
            del self.all_docs[title]
            return title
        return None

    def run_remove_page_if_few_references(self):
        pages_with_few_references = []
        keys = list(self.all_docs.keys())
        for key in tqdm.tqdm(keys):
            removed_doc = self.remove_page_if_few_references(key)
            if removed_doc is not None:
                pages_with_few_references.extend(removed_doc)
        self.pages_with_few_references.extend(pages_with_few_references)

    def check_for_zeros(self):
        for doc in tqdm.tqdm(self.all_docs.values()):
            if len(doc['refs']) == 0:
                return False
        return True

    def reuse_embeddings(self):
        print('loading embeddings')
        with open("data/full_wiki_with_embeddings.pickle", "rb") as handle:
            wiki_with_embeddings = pickle.load(handle)

        all_removed_keys = np.setdiff1d(
            list(wiki_with_embeddings.keys()), list(self.all_docs.keys())
        )
        for key in list(all_removed_keys):
            del wiki_with_embeddings[key]

        with open("data/wiki_with_embeddings.pickle", "wb+") as handle:
            pickle.dump(wiki_with_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_extractor = DataExtractor()
    data_extractor.run_extract_data()
    data_extractor.run_check_references(True)

    stable = False

    eps = 1000

    while not stable:
        length_before = len(data_extractor.all_docs)
        data_extractor.run_remove_page_if_few_references()
        data_extractor.run_check_references(False)
        length_after = len(data_extractor.all_docs)

        stable = (
            length_before - eps / 2 <= length_after <= length_before + eps / 2
        ) and data_extractor.check_for_zeros()

    print('saving wiki')
    with open("data/wiki.pickle", "wb+") as handle:
        pickle.dump(data_extractor.all_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('saving missed refs')
    with open("data/missed_refs.pickle", "wb+") as handle:
        pickle.dump(
            data_extractor.all_missed_refs, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    print('saving removed refs')
    with open("data/removed_refs.pickle", "wb+") as handle:
        pickle.dump(
            data_extractor.pages_with_few_references,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
