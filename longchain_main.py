import pickle

import faiss
from langchain.prompts import PromptTemplate
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

import defaults
import utils



defaults.load_creds()

index = faiss.read_index(defaults.LONGCHAIN_INDEX_FILE)
store = pickle.load(open(defaults.LONGCHAIN_DATA_STORE, "rb"))

questions = utils.load_questions()
questions = questions[0:5]

store.index = index

my_prompt = utils.longchain_prompt
my_prompt = utils.LONGCHAIN_QA_PROMPT_TMPL

c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=my_prompt)

chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store, combine_prompt=c_prompt)


def sources_to_url(answer):
    loc = answer.rfind("Sources:")
    if loc == -1:
        return answer, []
    sources = answer[loc + len("Sources:"):].strip().split(" and ")
    answer = answer[:loc].strip()
    return answer, sources

def result_to_str(ans):
    question = ans["question"]
    response = ans["answer"]
    answer, sources = sources_to_url(response)
    result = f'<div style="q">Q:{question}</div>\n'
    result += f'<div style="a">A:{answer}</div>\n'
    for url in sources:
        result += f'<a href="{url}">{url}</a>\n'
    return result


html_result = ""
for question in questions:
    chain_result = chain({"question": question})
    html_result += result_to_str(chain_result)

with open("longchain_result.html", "w") as f:
    f.write(f"<html><head><style>{utils.HTML_STYLE}</style></head><body>")
    f.write(html_result)
    f.write("</body></html>")

