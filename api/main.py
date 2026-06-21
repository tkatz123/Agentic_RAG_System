from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import build_graph


app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(request: QuestionRequest) -> AnswerResponse:
    agent = build_graph()
    result = agent.invoke({
        "question": request.question,
        "query": request.question,
        "docs": [],
        "relevant": False,
        "answer": ""
    })

    sources = []

    for doc in result['docs']:
        if doc.metadata.get("source") == "job_posting":
            sources.append(doc.metadata['company'])
        else:
            sources.append(doc.metadata['title'])
    
    sources = list(set(sources))

    return AnswerResponse(answer = result['answer'], sources = sources)