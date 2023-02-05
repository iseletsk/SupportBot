from gpt_index import GPTSimpleVectorIndex
import defaults

from utils import doc_id_to_url, MY_QA_PROMPT, MY_REFINE_PROMPT, HTML_STYLE, load_questions

defaults.load_creds()

vector_index = GPTSimpleVectorIndex.load_from_disk(defaults.VECTOR_INDEX_FILE, text_qa_template=MY_QA_PROMPT)
# list_index = GPTListIndex.load_from_disk(defaults.LIST_INDEX_FILE)


questions = load_questions()

# questions = questions[1:2]
print(f'Loaded {len(questions)} questions')

vector_result = ""
list_result = ""
count = 0

def result_to_str(question, response, index):
    result = f'<div style="q">Q:{question}</div>\n'
    result += f'<div style="a">A:{response.response}</div>\n'
    for node in response.source_nodes:
        doc = index.docstore.get_document(node.doc_id)
        url, prefix, internal = doc_id_to_url(doc.doc_id)
        result += f'<div class="d {prefix} {internal}">{doc.text}<br><a href="{url}">{url}</a></div>\n'
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

vector_index.save_to_disk(defaults.VECTOR_INDEX_FILE)

