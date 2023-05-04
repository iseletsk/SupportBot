from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

TICKETS = None

my_list = []


def init_tickets():
    global TICKETS
    if TICKETS is None:
        TICKETS = []
        with open('workdir/tickets_list.jsonl', 'r') as f:
            for line in f:
                element = json.loads(line)
                TICKETS.append(element)
    return TICKETS


@app.get("/tickets")
async def get_tickets():
    init_tickets()
    return TICKETS


