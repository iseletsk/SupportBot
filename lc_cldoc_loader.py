from typing import List, Optional
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from doc_loaders import init_docs
from utils import doc_id_to_url

class CLDocLoader(BaseLoader):
    """Load CLdoc files."""

    def __init__(self, descriptor: dict, update_git: bool, encoding: Optional[str] = None):
        """Initialize with file path."""
        self.descriptor = descriptor
        self.update_git = update_git
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        llama_index_docs = init_docs(self.descriptor, self.update_git)
        docs = []
        for doc in llama_index_docs:
            metadata = {"source": doc_id_to_url(doc.doc_id)[0]}
            docs.append(Document(page_content=doc.text, metadata=metadata))
        return docs


class KBDocLoader(BaseLoader):
    """Load KBdoc files."""

    def __init__(self, path: str, strip_html: bool = True, encoding: Optional[str] = None):
        """Initialize with file path."""
        self.path = path
        self.strip_html = strip_html
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        import re
        import json
        with open(self.path, 'r') as f:
            tag_re = re.compile(r'<[^>]+>')
            kb = json.load(f)
            documents = []
            for article in kb:
                body = article['body']
                if self.strip_html:
                    body = tag_re.sub('', body)
                body = article['title'] + '\n\n' + body
                url = article['html_url']
                metadata ={"source": url,
                           "labels": ",".join(article["labels"]).lower()}
                documents.append(Document(page_content=body, metadata=metadata))
            return documents
