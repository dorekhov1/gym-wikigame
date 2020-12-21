import re
import pickle

from tqdm import tqdm


class DataExtractor:

    pages_title_regex = r"[0-9]+:([0-9]+):(.*)"

    pages_row_regex = r"\((.*?,.*?,'.*?',.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?,.*?)\)[,;]"
    pages_list_regex = r"([0-9]*),([-+]?[0-9]*),'(.*?)',(.*?),([01]),([01]),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*)"

    redirects_row_regex = r"\((.*?,.*?,'.*?','.*?','.*?')\)[,;]"
    redirects_list_regex = r"([0-9]*),([-+]?[0-9]*),'(.*?)',('.*?'),('.*')"

    link_row_regex = r"\((.*?,.*?,'.*?',.*?)\)[,;]"
    link_list_regex = r"([0-9]*),([-+]?[0-9]*),'(.*?)',(.*)"

    def __init__(self):
        self.pages = {}
        self.title_to_id_map = {}

    def process_pages(self):
        file_path = "data/raw/enwiki-20201020-page.sql"
        num_lines = sum(1 for _ in open(file_path, "rt"))
        with open(file_path, "rt") as f:
            for line in tqdm(f, total=num_lines):
                all_rows = re.findall(DataExtractor.pages_row_regex, line)
                for row in all_rows:
                    row_list = re.findall(DataExtractor.pages_list_regex, row)[0]
                    if row_list[1] == "0":
                        doc = {"id": row_list[0], "title": row_list[2], "links": set()}
                        self.pages[doc["id"]] = doc
                        self.title_to_id_map[doc["title"]] = doc["id"]

    def process_redirects(self):
        file_path = "data/raw/enwiki-20201020-redirect.sql"
        num_lines = sum(1 for _ in open(file_path, "rt"))

        missed_redirects = 0
        with open(file_path, "rt") as f:
            for line in tqdm(f, total=num_lines):
                all_rows = re.findall(DataExtractor.redirects_row_regex, line)
                for row in all_rows:
                    row_list = re.findall(DataExtractor.redirects_list_regex, row)[0]
                    redirect_id = row_list[0]
                    redirect_target = row_list[2]

                    try:
                        target_page = self.pages[self.title_to_id_map[redirect_target]]
                    except:
                        missed_redirects += 1
                        continue

                    self.pages[redirect_id] = target_page
        print(f"Missed redirects: {missed_redirects}")

    def process_links(self):
        file_path = "data/raw/enwiki-20201020-pagelinks.sql"
        num_lines = sum(
            1 for _ in open(file_path, "rt", encoding="utf-8", errors="ignore")
        )

        total_links = 0
        missed_links_by_title = 0
        missed_links_by_link_id = 0
        with open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, total=num_lines):
                all_rows = re.findall(DataExtractor.link_row_regex, line)
                for row in all_rows:
                    row_list = re.findall(DataExtractor.link_list_regex, row)[0]
                    link_id = row_list[0]
                    link_target = row_list[2]

                    # need to do this in case the target is a redirect
                    if link_target in self.title_to_id_map:
                        target_id = self.title_to_id_map[link_target]
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
