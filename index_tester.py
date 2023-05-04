from llama_index import GPTSimpleVectorIndex
from llama_index.docstore import DocumentStore


import defaults

from utils import doc_id_to_url, MY_QA_PROMPT, MY_REFINE_PROMPT, HTML_STYLE, load_questions
from doc_loaders import load_index, save_index

defaults.load_creds()

vector_index = load_index(defaults.VECTOR_INDEX_FILE)
vector_index.text_qa_template = MY_QA_PROMPT
# vector_index = GPTSimpleVectorIndex.load_from_disk(defaults.VECTOR_INDEX_FILE,
#                                                    text_qa_template=MY_QA_PROMPT)
# list_index = GPTListIndex.load_from_disk(defaults.LIST_INDEX_FILE)


questions = load_questions()

questions = questions[0:10]
print(f'Loaded {len(questions)} questions')

vector_result = ""
list_result = ""
count = 0

def result_to_str(question, response, index):
    result = f'<div style="q">Q:{question}</div>\n'
    result += f'<div style="a">A:{response.response}</div>\n'
    for node in response.source_nodes:
        # doc = index.docstore.get_document(node.node.doc_id)
        url, prefix, internal = doc_id_to_url(node.node.ref_doc_id)
        result += f'<div class="d {prefix} {internal}">{node.node.get_text()}<br><a href="{url}">{url}</a></div>\n'
    return result


for q in questions:
    try:
        response = vector_index.query(q, text_qa_template=MY_QA_PROMPT, refine_template=MY_REFINE_PROMPT,
                                      similarity_top_k=3)
        vector_result += result_to_str(q, response, vector_index)
        count += 1
        print(f'Vector processed {count} questions out of {len(questions)}')
    except Exception as e:
        import traceback
        traceback.print_exc()

with open("vector_result.html", "w") as f:
    f.write(f"<html><head><style>{HTML_STYLE}</style></head><body>")
    f.write(vector_result)
    f.write("</body></html>")

save_index(vector_index, defaults.VECTOR_INDEX_FILE)

