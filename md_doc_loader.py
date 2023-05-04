from llama_index import GPTSimpleVectorIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.data_structs.node_v2 import DocumentRelationship

import defaults
from defaults import kb_articles
from doc_loaders import init_docs
from doc_loaders import load_kb_json
from doc_loaders import save_index
import pickle



UPDATE_FROM_GIT = False

cl_docs = init_docs(defaults.cloudlinux_docs, update_git=UPDATE_FROM_GIT)
im_docs = init_docs(defaults.imunify_docs, update_git=UPDATE_FROM_GIT)
kb_docs = load_kb_json(kb_articles, strip_html=True)
print(len(cl_docs), len(im_docs), len(kb_docs))


docs = cl_docs + im_docs + kb_docs
#docs = cl_docs[2:15]
print(f'Loaded {len(docs)} documents')

defaults.load_creds()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(docs)
vector_index = GPTSimpleVectorIndex(nodes)


#list_index = GPTListIndex(docs)

save_index(vector_index, defaults.VECTOR_INDEX_FILE)

