import pickle

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import defaults
import utils
import doc_loaders

defaults.load_creds()

UPDATE_FROM_GIT = False

cl_docs = doc_loaders.init_docs(defaults.cloudlinux_docs, update_git=UPDATE_FROM_GIT)
im_docs = doc_loaders.init_docs(defaults.imunify_docs, update_git=UPDATE_FROM_GIT)
kb_docs = doc_loaders.load_kb_json(defaults.kb_articles, strip_html=True)
print(len(cl_docs), len(im_docs), len(kb_docs))

docs = cl_docs + im_docs + kb_docs

data = []
sources = []
for p in docs:
    data.append(p.text)
    sources.append(utils.doc_id_to_url(p.doc_id))

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, defaults.LONGCHAIN_INDEX_FILE)
store.index = None
with open(defaults.LONGCHAIN_DATA_STORE, "wb") as f:
    pickle.dump(store, f)