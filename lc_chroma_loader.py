from typing import List, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import defaults

import doc_loaders
from lc_cldoc_loader import CLDocLoader, KBDocLoader

import os
import shutil


def _persist_dir(descriptor:dict, persist_dir: str) -> str:
    return f'{persist_dir}/{descriptor["prefix"]}'


def viewpress2chroma(descriptor:dict, update_git: bool, remove_old: bool, persist_dir: str) -> Chroma:
    """ Load documents from git repo and save them to Chroma vector store."""
    # https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html
    embeddings = OpenAIEmbeddings()

    loader = CLDocLoader(descriptor, update_git=update_git)
    viewpress_docs = loader.load()

    persist_dir = _persist_dir(descriptor, persist_dir)
    if remove_old:
        shutil.rmtree(persist_dir, ignore_errors=True)

    os.makedirs(persist_dir, exist_ok=True)

    vectordb = Chroma.from_documents(viewpress_docs, embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb


def load_chroma(descriptor: dict, persist_dir: str) -> Chroma:
    persist_directory = _persist_dir(descriptor, persist_dir)
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    descriptor['vectordb'] = db
    return db

def kb2chroma(path, cloudlinux_db, imunify_db):
    loader = KBDocLoader(path=path, strip_html=True)
    kb_docs = loader.load()
    cldocs = []
    imdocs = []
    for doc in kb_docs:
        labels = doc.metadata['labels']
        if 'imunify' in labels:
            imdocs.append(doc)
        else:
            cldocs.append(doc)
    if imdocs:
        texts = [doc.page_content for doc in imdocs]
        metadatas = [doc.metadata for doc in imdocs]
        imunify_db.add_texts(texts, metadatas)
    if cldocs:
        texts = [doc.page_content for doc in cldocs]
        metadatas = [doc.metadata for doc in cldocs]
        cloudlinux_db.add_texts(texts, metadatas)
    cloudlinux_db.persist()
    imunify_db.persist()


def init_chroma_dbs(update_from_git: bool, persist_dir: str):
    import defaults
    import shutil
    shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)
    cln_db = viewpress2chroma(defaults.cln_docs, update_git=update_from_git, remove_old=True, persist_dir=persist_dir)
    tuxcare_db = viewpress2chroma(defaults.tuxcare_docs, update_git=update_from_git, remove_old=True, persist_dir=persist_dir)
    cloudlinux_db = viewpress2chroma(defaults.cloudlinux_docs, update_git=update_from_git, remove_old=True, persist_dir=persist_dir)
    imunify_db = viewpress2chroma(defaults.imunify_docs, update_git=update_from_git, remove_old=True, persist_dir=persist_dir)
    kb2chroma(defaults.kb_articles, cloudlinux_db, imunify_db)
    return cloudlinux_db, imunify_db, cln_db, tuxcare_db


def load_chroma_dbs(persist_dir: str) -> List[Dict]:
    cloudlinux_db = load_chroma(defaults.cloudlinux_docs, persist_dir)
    imunify_db = load_chroma(defaults.imunify_docs, persist_dir)
    cln_db = load_chroma(defaults.cln_docs, persist_dir)
    tuxcare_db = load_chroma(defaults.tuxcare_docs, persist_dir)

    #defaults.cloudlinux_docs['vectordb'] = cloudlinux_db
    #defaults.imunify_docs['vectordb'] = imunify_db
    return [defaults.cloudlinux_docs, defaults.imunify_docs, defaults.cln_docs, defaults.tuxcare_docs]


