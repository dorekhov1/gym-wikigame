import re
import os
import pickle
import json

from urllib.parse import unquote
from tqdm import tqdm
from multiprocessing import Pool


class DataExtractor:

    extracted_dir = "data/raw/extracted"
    pages_file_path = "data/raw/enwiki-20210301-page.sql"
    redirect_file_path = "data/raw/enwiki-20210301-redirect.sql"
    links_file_path = "data/raw/enwiki-20210301-pagelinks.sql"

    pages_title_regex = r"[0-9]+:([0-9]+):(.*)"

    pages_row_regex = r"\((.*?,.*?,'.*?',.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?)\)[,;]"
    pages_list_regex = r"([0-9]*),([-+]?[0-9]*),'(.*?)',(.*?),([01]),([01]),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*)"

    redirects_row_regex = r"\((.*?,.*?,'.*?','.*?','.*?')\)[,;]"
    redirects_list_regex = r"([0-9]*),([-+]?[0-9]*),'(.*?)',('.*?'),('.*')"

    link_row_regex = r"\((.*?,.*?,'.*?',.*?)\)[,;]"
    link_list_regex = r"([0-9]*),([-+]?[0-9]*),'(.*?)',(.*)"

    href_regex = r"&lt;a href=\"(.*?)\".*?&lt;\/a&gt;"

    def __init__(self):
        self.pages = {}
        self.title_to_id_map = {}

    def process_pages(self):
        num_lines = sum(1 for _ in open(DataExtractor.pages_file_path, "rt"))
        with open(DataExtractor.pages_file_path, "rt") as f:
            for line in tqdm(f, total=num_lines):
                all_rows = re.findall(DataExtractor.pages_row_regex, line)
                for row in all_rows:
                    row_list = re.findall(DataExtractor.pages_list_regex, row)[0]
                    if row_list[1] == "0":
                        normalized_title = DataExtractor.normalize_title(row_list[2])
                        doc = {"id": row_list[0], "title": normalized_title, "links": set()}
                        self.pages[doc["id"]] = doc
                        self.title_to_id_map[normalized_title] = doc["id"]

    def process_redirects(self):
        num_lines = sum(1 for _ in open(DataExtractor.redirect_file_path, "rt"))

        missed_redirects = 0
        with open(DataExtractor.redirect_file_path, "rt") as f:
            for line in tqdm(f, total=num_lines):
                all_rows = re.findall(DataExtractor.redirects_row_regex, line)
                for row in all_rows:
                    row_list = re.findall(DataExtractor.redirects_list_regex, row)[0]
                    redirect_id = row_list[0]
                    redirect_target = DataExtractor.normalize_title(row_list[2])

                    try:
                        target_page = self.pages[self.title_to_id_map[redirect_target]]
                    except:
                        missed_redirects += 1
                        continue

                    self.pages[redirect_id] = target_page
        print(f"Missed redirects: {missed_redirects}")

    def process_links_from_xml(self):

        def files_generator(data_dir):
            for directory in os.listdir(data_dir):
                dir_path = os.path.join(data_dir, directory)
                if os.path.isdir(dir_path):
                    for file in os.listdir(dir_path):
                        if file.startswith("wiki_"):
                            yield os.path.join(dir_path, file)

        files = list(files_generator(DataExtractor.extracted_dir))

        total_links = 0
        missed_links_by_title = 0
        missed_links_by_link_id = 0
        for file_path in tqdm(files):
            with open(file_path) as f:
                for line in f:
                    wiki_doc = json.loads(line)
                    link_id = wiki_doc['id']
                    proper_title = wiki_doc['title']
                    normalized_title = DataExtractor.normalize_title(proper_title)
                    if link_id not in self.pages:
                        continue

                    if normalized_title == self.pages[link_id]['title']:
                        self.pages[link_id]['proper_title'] = proper_title

                    refs = [
                        DataExtractor.normalize_title(unquote(ref))
                        for ref in re.findall(self.href_regex, wiki_doc["text"])
                    ]

                    for ref in refs:
                        # need to do this in case the target is a redirect
                        if ref in self.title_to_id_map:
                            target_id = self.title_to_id_map[ref]
                            target_title = self.pages[target_id]["title"]
                        else:
                            missed_links_by_title += 1
                            continue

                        if link_id in self.pages and target_title != self.pages[link_id]['title']:
                            self.pages[link_id]["links"].add(target_title)
                        else:
                            missed_links_by_link_id += 1
                            continue

                        total_links += 1

        print(f"Missed links by title: {missed_links_by_title}")
        print(f"Missed links by link id: {missed_links_by_link_id}")
        print(f"Total links: {total_links}")

    @staticmethod
    def normalize_title(title: str):
        title = title.replace(" ", "")
        title = title.replace("_", "")
        title = title.lower()
        return title

    def remove_redirect_pages(self):
        redirect_ids = []
        for page_id in tqdm(self.pages.keys()):
            if page_id != self.pages[page_id]["id"]:
                redirect_ids.append(page_id)

        for page_id in tqdm(redirect_ids):
            del self.pages[page_id]

    def change_keys_to_title(self):
        all_ids = list(self.pages.keys())
        for page_id in tqdm(all_ids):
            page = self.pages[page_id]
            title = page["title"]
            self.pages[title] = self.pages.pop(page_id)

    def remove_pages_with_few_links(self, n=500, recursive=False):
        num_pages = len(self.pages)

        pages_to_remove = []
        for page_title in tqdm(self.pages.keys()):
            page = self.pages[page_title]
            page_links = page['links']

            if len(page_links) < n:
                pages_to_remove.append(page_title)

        for page_title in pages_to_remove:
            del self.pages[page_title]

        self.remove_invalid_links()

        if recursive and len(self.pages) != num_pages:
            self.remove_pages_with_few_links()

    def remove_invalid_links(self):
        num_links_removed = 0
        all_titles = set(self.pages.keys())
        for page_title in tqdm(all_titles):
            page = self.pages[page_title]
            page_links = page['links']

            links_to_remove = []
            for link in page_links:
                if link not in all_titles:
                    num_links_removed += 1
                    links_to_remove.append(link)

            for link in links_to_remove:
                page_links.remove(link)

        print(f"Number of links removed {num_links_removed}")

    def save_pages(self):
        with open("data/processed/pages.pickle", "wb+") as handle:
            pickle.dump(
                self.pages, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open("data/processed/title_to_id_map.pickle", "wb+") as handle:
            pickle.dump(
                self.title_to_id_map, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_pages(self):
        with open("data/processed/pages.pickle", "rb") as handle:
            self.pages = pickle.load(handle)

        with open("data/processed/title_to_id_map.pickle", "rb") as handle:
            self.title_to_id_map = pickle.load(handle)


if __name__ == "__main__":
    data_extractor = DataExtractor()

    print("Processing pages")
    data_extractor.process_pages()

    print("Processing redirects")
    data_extractor.process_redirects()

    print("Processing links")
    data_extractor.process_links_from_xml()

    print("Removing redirect pages")
    data_extractor.remove_redirect_pages()

    print("Changing keys")
    data_extractor.change_keys_to_title()

    print("Removing invalid links")
    data_extractor.remove_invalid_links()

    print("Removing pages with few links")
    data_extractor.remove_pages_with_few_links(n=100)

    print("Saving pages")
    data_extractor.save_pages()
