from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag.chain import answer_question

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/api/ask")
def ask(q: Query):
    return {"answer": answer_question(q.query)}

# Serve the HTML frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")