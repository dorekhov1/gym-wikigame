import pickle

print('loading embeddings')
with open("data/full_wiki_with_embeddings.pickle", "rb") as handle:
    wiki_with_embeddings = pickle.load(handle)

print('loading wiki')
with open("data/wiki.pickle", "rb") as handle:
    all_docs = pickle.load(handle)

print('removing')
for doc_key in all_docs.keys():
    all_docs[doc_key]['title_embedding'] = wiki_with_embeddings[doc_key]['title_embedding']

print('saving result')
with open("data/wiki_with_embeddings.pickle", "wb+") as handle:
    pickle.dump(all_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
