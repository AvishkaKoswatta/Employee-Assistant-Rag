# api.py
from fastapi import FastAPI
from pydantic import BaseModel

from rag.retriever import load_vectorstore
from rag.chain import get_rag_answer

app = FastAPI()

# Load once (important 🚨)
vectorstore = load_vectorstore()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = get_rag_answer(request.query, vectorstore)
    return {
        "query": request.query,
        "answer": answer
    }