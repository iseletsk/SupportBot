from gpt_index import GPTSimpleVectorIndex

import defaults
from defaults import kb_articles
from doc_loaders import init_docs
from doc_loaders import load_kb_json


UPDATE_FROM_GIT = False

cl_docs = init_docs(defaults.cloudlinux_docs, update_git=UPDATE_FROM_GIT)
im_docs = init_docs(defaults.imunify_docs, update_git=UPDATE_FROM_GIT)
kb_docs = load_kb_json(kb_articles, strip_html=True)
print(len(cl_docs), len(im_docs), len(kb_docs))


docs = cl_docs + im_docs + kb_docs
print(f'Loaded {len(docs)} documents')

defaults.load_creds()
vector_index = GPTSimpleVectorIndex(docs)
#list_index = GPTListIndex(docs)

vector_index.save_to_disk(defaults.VECTOR_INDEX_FILE)




