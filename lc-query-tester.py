import langchain

import defaults
from utils import load_questions
from lc_chroma_loader import load_chroma_dbs, init_chroma_dbs

from vector_store_qa import init_retrieval_vectorstore_agent, init_original_vectorstore_agent

UPDATE_FROM_GIT = False
PERSIST_DIR = 'chroma'
VERBOSE = True
langchain.verbose = VERBOSE

defaults.load_creds()

# init_chroma_dbs(UPDATE_FROM_GIT, PERSIST_DIR)
docs = load_chroma_dbs(persist_dir=PERSIST_DIR)

# agent_executor = init_vectorstore_agent(docs, verbose=VERBOSE)
#agent_executor = init_original_vectorstore_agent(docs, verbose=VERBOSE)
my_qa_tools = None
agent_executor, my_qa_tools = init_retrieval_vectorstore_agent(docs, verbose=VERBOSE)

questions = load_questions()
questions = questions[0:5]
for question in questions:
    print(f"Question: {question}")
    response = agent_executor({
        "input": question,
    })
    print(f"Response: {response['output']}")
    if my_qa_tools:
        for tool in my_qa_tools:
            for source in tool.sources:
                meta = source.metadata
                print(f"   {meta['source']}")
            tool.reset()
