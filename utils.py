import csv
import json

from gpt_index import QuestionAnswerPrompt, RefinePrompt

import defaults

KB_DOCS = None


def get_kb_docs():
    global KB_DOCS
    if KB_DOCS is None:
        KB_DOCS = {}
        with open(defaults.KB_JSON_FILE, 'r') as f:
            data = json.load(f)
            for article in data:
                KB_DOCS[str(article['id'])] = article
    return KB_DOCS


def doc_id_to_url(doc_id):
    url = None
    prefix = None
    internal = False
    if doc_id.startswith('imunify.'):
        url = f'https://docs.imunify360.com{doc_id[len("imunify."):]}'
        prefix = 'imunify'
    elif doc_id.startswith('cloudlinux.'):
        url = f'https://docs.cloudlinux.com{doc_id[len("cloudlinux."):]}'
        prefix = 'cloudlinux'
    elif doc_id.startswith('KB.'):
        kb_id = doc_id[len("KB."):]
        article = get_kb_docs()[kb_id]
        url = article['html_url']
        prefix = f'KB'
        internal = article['internal']
    return url, prefix, internal




QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
    "If you don't know the answer, just say \"I don't know\"\n"
)
MY_QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
MY_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "The context information is below. \n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "We have provided an existing answer: {existing_answer}\n"
    "If the existing answer is \"I don't known\", given the context information and not prior knowledge, "
    "answer the question. If you don't know the answer, just say \"I don't know\"\n"
    "Otherwise have the opportunity to refine the existing answer"
    "(only if needed) given the context information above.\n"
    "Given the new information, refine the original answer to better answer the question. "
    "If the context isn't useful return the original answer."
)
MY_REFINE_PROMPT = RefinePrompt(MY_REFINE_PROMPT_TMPL)

longchain_prompt = """
You are a CloudLinux bot assistant helping customers. You give short answers.
Given no prior knowledge, the following extracted parts of a long document and a question, 
create a final answer with references ("SOURCES").
Answer "I don't know" if you don't know the answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: How to merge tables in pandas?
=========
Content: pandas provides various facilities for easily combining together Series or DataFrame with various kinds of set logic for the indexes and relational algebra functionality in the case of join / merge-type operations.
Source: 28-pl
Content: pandas provides a single function, merge(), as the entry point for all standard database join operations between DataFrame or named Series objects: \n\npandas.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
Source: 30-pl
=========
FINAL ANSWER: To merge two tables in pandas, you can use the pd.merge() function. The basic syntax is: \n\npd.merge(left, right, on, how) \n\nwhere left and right are the two tables to merge, on is the column to merge on, and how is the type of merge to perform. \n\nFor example, to merge the two tables df1 and df2 on the column 'id', you can use: \n\npd.merge(df1, df2, on='id', how='inner')
SOURCES: 28-pl 30-pl

QUESTION: How to eat vegetables using pandas?
=========
Content: ExtensionArray.repeat(repeats, axis=None) Returns a new ExtensionArray where each element of the current ExtensionArray is repeated consecutively a given number of times. \n\nParameters: repeats int or array of ints. The number of repetitions for each element. This should be a positive integer. Repeating 0 times will return an empty array. axis (0 or ‘index’, 1 or ‘columns’), default 0 The axis along which to repeat values. Currently only axis=0 is supported.
Source: 0-pl
=========
FINAL ANSWER: You can't eat vegetables using pandas. You can only eat them using your mouth.
SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:
"""

LONGCHAIN_QA_PROMPT_TMPL = """
Context information is below.
========
{summaries}
========
Given the context information and no prior knowledge,
answer the question: {question}
If you don't know the answer, say "I don't know"
Provide a short final answer and sources.
"""

HTML_STYLE = """
    div {margin: 10px 0;}
    div.q {font-weight: bold;}
    div.a {font-style: normal;}
    div.d {background-color: #f0f0f0; font-style: italic; }
    div.d.cloudlinux {background-color: #3293a8; }
    div.d.cloudlinux.True { border-color: red; border-width: 2px; border-style: solid; }
    div.d.cloudlinux.False { border-color: green; border-width: 2px; border-style: solid; }
    div.d.imunify { background-color: #53995d; }
    div.d.KB { background-color: #f0f0f0; }
    div.d.KB.True { border-color: red; border-width: 2px; border-style: solid; }
"""


def load_questions():
    questions = []
    with open("questions.csv", "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) > 0 and len(line[0]) > 0:
                questions.append(line[0])
    return questions
