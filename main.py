import os

from gpt_index import GPTSimpleVectorIndex
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import defaults
import utils

INDEX_FILE = defaults.VECTOR_INDEX_FILE


def load_data(load_from_disk=True):
    if load_from_disk and os.path.exists(INDEX_FILE):
        print('Loading index from disk...')
        return GPTSimpleVectorIndex.load_from_disk(INDEX_FILE)
    else:
        raise Exception('Index file not found', INDEX_FILE)

creds = defaults.load_creds()

index = load_data()
app = App(token=creds['slack-bot-token'])

@app.event("message")
def handle_message_im(payload, client):
    response = get_response(payload["text"])
    client.chat_postMessage(channel=payload['channel'], text=response)


@app.event("app_mention")
def handle_app_mentions(body, say, logger):
    question = body["event"]["text"]
    try:
        itself = body["authorizations"][0]["user_id"]
        question = question.replace(f'<@{itself}>', '')
    except Exception as e:
        print(e)
        pass
    response = get_response(question)
    say(response)

@app.command("/clthink")
def command_clthink(ack, respond, command):
    ack()
    question = command['text']
    print("question received:", question)
    response = get_response(question, top_k=3)
    print("response:", response)
    respond(f'Question: {question}\n\nAnswer: {response}\n\n')

@app.command("/cldoc")
def command_cldoc(ack, respond, command):
    ack()
    question = command['text']
    print("question received:", question)
    response = get_response(question)
    print("response:", response)
    respond(f'Question: {question}\n\nAnswer: {response}\n\n')


def response_formatting(response):
    result = f'{response.response}\n\n'
    for node in response.source_nodes:
        doc = index.docstore.get_document(node.doc_id)
        url, prefix, internal = utils.doc_id_to_url(doc.doc_id)
        result += '>' + doc.text.replace('\n', '\n>') + f'\n> <{url}>\n\n'
    return result

def get_response(question, top_k=1):
    response = index.query(question, text_qa_template=utils.MY_QA_PROMPT, refine_template=utils.MY_REFINE_PROMPT,
                           similarity_top_k=top_k)
    return response_formatting(response)


handler = SocketModeHandler(app, creds['slack-app-token'])
handler.start()