from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex
import os
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler


INDEX_FILE = 'workdir/index.json'


def load_creds():
    with open('credentials/access-tokens.json', 'r') as f:
        creds = json.load(f)
        return creds

def load_data(load_from_disk=True):
    if load_from_disk and os.path.exists(INDEX_FILE):
        print('Loading index from disk...')
        return GPTSimpleVectorIndex.load_from_disk(INDEX_FILE)
    documents = SimpleDirectoryReader('docs').load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(INDEX_FILE)
    return index

creds = load_creds()
os.environ["OPENAI_API_KEY"] = creds['openai_api_key']
index = load_data()
app = App(token=creds['slack-bot-token'])


@app.command("/cldoc")
def command_cldoc(ack, say, command):
    ack()
    print("question received:", command['text'])
    response = index.query(command['text'])
    print("response:", response)
    say(str(response))


handler = SocketModeHandler(app, creds['slack-app-token'])
handler.start()