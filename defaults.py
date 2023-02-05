import json
import os

WORK_DIR = 'workdir'
VECTOR_INDEX_FILE = f'{WORK_DIR}/vector_index.json'
LIST_INDEX_FILE = f'{WORK_DIR}/list_index.json'
KB_JSON_FILE = f'{WORK_DIR}/articles-docs.json'

LONGCHAIN_INDEX_FILE = f'{WORK_DIR}/longchain_index.json'
LONGCHAIN_DATA_STORE = f'{WORK_DIR}/longchain_data_store.pkl'

def load_creds():
    with open('credentials/access-tokens.json', 'r') as f:
        creds = json.load(f)
        os.environ["OPENAI_API_KEY"] = creds['openai_api_key']
        return creds

def save_index(vector_index, list_index):
    vector_index.save_to_disk(VECTOR_INDEX_FILE)
    list_index.save_to_disk(LIST_INDEX_FILE)


imunify_docs = {"git": "https://github.com/cloudlinux/imunify360-doc.git",
                "dir": "imunify360-doc",
                "url": "https://docs.imunify360.com/",
                "prefix": "imunify"}

cloudlinux_docs = {
    "git": "https://github.com/cloudlinux/cloudlinux-doc.git",
    "dir": "cloudlinux-doc",
    "url": "https://docs.cloudlinux.com/",
    "prefix": "cloudlinux"
}
kb_articles = '/Users/iseletsk/lve/randomHelpers/data/zendesk-kb-export/articles-docs.json'

