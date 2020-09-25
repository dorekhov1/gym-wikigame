import os
import re
import json
import pickle
from urllib.parse import unquote
from multiprocessing import Pool

import tqdm as tqdm


class DataExtractor:
    def __init__(self):
        self.data_dir = "data/extracted"
        self.href_regex = r"<a href=\"(.*?)\".*?<\/a>"

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
                del wiki_doc["title"]
                del wiki_doc["url"]

        return data


if __name__ == "__main__":
    data_extractor = DataExtractor()
    files = list(data_extractor.files_generator())

    all_docs = {}
    with Pool(processes=15) as p:
        with tqdm.tqdm(total=len(files)) as pbar:
            for i, docs in enumerate(
                p.imap_unordered(data_extractor.extract_data, files)
            ):
                all_docs.update(docs)
                pbar.update()

    with open("data/wiki.pickle", "wb+") as handle:
        pickle.dump(all_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
