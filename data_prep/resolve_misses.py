import pickle
from urllib.parse import urlparse

import mwclient
import tqdm


def url_validator(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


with open("../data/missed_refs.pickle", "rb") as handle:
    missed_refs = pickle.load(handle)

with open("../data/wiki.pickle", "rb") as handle:
    wiki = pickle.load(handle)

site = mwclient.Site('en.wikipedia.org')

resolved_refs = {}
broken_refs = []
still_broken = []
for missed_ref in tqdm.tqdm(missed_refs):
    if url_validator(missed_ref):
        broken_refs.append(missed_ref)
        continue
    try:
        page = site.pages[missed_ref]
        if not page.exists:
            still_broken.append(missed_ref)
            continue

        if page.redirect:
            redirect_page = page.redirects_to()
            resolved_refs[missed_ref] = redirect_page.base_name
        else:
            resolved_refs[missed_ref] = page.base_name

        if resolved_refs[missed_ref] not in wiki.keys():
            still_broken.append(resolved_refs[missed_ref])
    except:
        broken_refs.append(missed_ref)

with open("data/resolved_refs.pickle", "wb+") as handle:
    pickle.dump(resolved_refs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/broken_refs.pickle", "wb+") as handle:
    pickle.dump(broken_refs, handle, protocol=pickle.HIGHEST_PROTOCOL)