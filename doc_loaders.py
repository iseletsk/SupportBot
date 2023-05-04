import json
import os
import re

from llama_index import Document
from llama_index.readers.file.markdown_parser import MarkdownParser
from llama_index import GPTSimpleVectorIndex

from slugify import slugify
import shutil
import pickle

WORKDIR = 'workdir-1'


def init_docs(settings, update_git=True):
    current = os.getcwd()
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)
    if update_git:
        shutil.rmtree(settings['dir'], ignore_errors=True)
        os.system(f'git clone --depth=1 {settings["git"]}')
        shutil.rmtree(f'{settings["dir"]}/docs/ru', ignore_errors=True)  # remove russian docs

    os.chdir(settings['dir'])
    docs = load_md_dirs('docs', settings['prefix'])
    os.chdir(current)
    return docs


def load_md_dirs(dir, prefix):
    documents = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.md'):
                documents += load_md_file(os.path.join(root, file), f'{prefix}.{root[len(dir):]}')
    return documents


def cleanup_notranslate(text):
    return text.replace('<div class="notranslate">', '').replace('</div>', '') \
        .replace('<span class="notranslate">', '').replace('</span>', '')


def load_md_file(path, prefix):
    documents = []
    md_parser = MarkdownParser(remove_hyperlinks=True, remove_images=True)
    docs = md_parser.parse_tups(path)
    if not docs[0][1]:
        return documents
    for header, text in docs:
        if not header:
            header = ""
        if not text:
            continue
        header = cleanup_notranslate(header)
        slug = slugify(header)
        text = f'{header}\n\n{cleanup_notranslate(text)}'
        d = Document(text)
        d.doc_id = f'{prefix}/#{slug}'
        documents.append(d)
    return documents


def load_kb_json(path, strip_html=True):
    with open(path, 'r') as f:
        tag_re = re.compile(r'<[^>]+>')
        kb = json.load(f)
        documents = []
        for article in kb:
            body = article['body']
            if strip_html:
                body = tag_re.sub('', body)
            body = article['title'] + '\n\n' + body
            d = Document(body)
            d.doc_id = f'KB.{article["id"]}'
            documents.append(d)
        return documents

def save_index(index, path):
    index.save_to_disk(path)
    with open(f'{path}.ds', "wb") as f:
        pickle.dump(index.docstore, f)


def load_index(path):
    with open(f'{path}.ds', "rb") as f:
        ds = pickle.load(f)
        ##index = GPTSimpleVectorIndex.load_from_disk(path, docstore=ds)
        index = GPTSimpleVectorIndex.load_from_disk(path)
        return index

