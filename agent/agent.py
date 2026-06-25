from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from openai import OpenAI
import os

load_dotenv()

class AgentState(TypedDict):
    question: str
    query: str
    docs: List[Document]
    relevant: bool
    answer: str
    retries: int

def retrieve(state: AgentState) -> dict:
    
    #Loading database object
    db = Chroma(persist_directory = "data/chroma_db", embedding_function = OpenAIEmbeddings())

    #Retriving top 10 relevant chunks
    docs = db.similarity_search(state['query'], k = 10)

    return {"docs": docs}

def grade_chunks(state: AgentState) -> dict:

    context = "\n\n".join([doc.page_content for doc in state['docs']])

    input_string = f"""
        You are a helpful assistant that determines if the context is relevant to answering the question.

        Context:
        {context}

        Question:
        {state['query']}

        Output:
        You can only return one of two values: True or False. Return True if the context is relevant and useful to answer the question, return False if the context is not relevant and useful to answer the question. 

        DO NOT RETURN ANYTHING ELSE BESIDES True OR False
    """

    #Initlaizing LLM object
    client = OpenAI()

    #Calls the LLM and inputs the prompt above
    response = client.responses.create(
        model = "gpt-4o-mini", 
        input = input_string
    )

    #Checks LLM response, returns associated boolean value
    if response.output_text.strip().lower() == 'true':
        return {'relevant': True}
    elif response.output_text.strip().lower() == 'false':
        return {'relevant': False}
    else:
        return {'relevant': False}
    

def reformulate(state: AgentState) -> dict:

    input_string = f"""
    You are a helpful assistant that helps rewrite queries to produce better results in a RAG pipeline. Below is the original question and the query that didn't work well, pelase rewrite the query to find better results.

    Question:
    {state['question']}

    Query:
    {state['query']}
    """

    #Initializes LLM object
    client = OpenAI()

    #Calls the LLM and inputs the prompt above
    response = client.responses.create(
        model = "gpt-4o-mini", 
        input = input_string
    )

    retries_count = state['retries'] + 1

    return {'query': response.output_text, 'retries': retries_count}

def generate(state: AgentState) -> dict:

    #Combining chunks into single string
    context = "\n\n".join([doc.page_content for doc in state["docs"]])

    #Creates stuffing prompt to feed to LLM
    input_string = f"""You are a helpful assistant. Use the context below to answer the question. If the answer is not in the context, say so

    Context:
    {context}

    Question: {state['question']}
    Answer:

    """

    #Initialize OpenAI object
    client = OpenAI()

    #Sends query to LLM
    response = client.responses.create(
        model = "gpt-4o-mini", 
        input = input_string
        )

    return {'answer': response.output_text}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_chunks", grade_chunks)
    graph.add_node("reformulate", reformulate)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges(
        "grade_chunks",
        lambda state: "generate" if state["relevant"] or state["retries"] >= 2 else "reformulate"
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()

if __name__ == "__main__":

    agent = build_graph()

    result = agent.invoke({
        "question": "What Python skills do AI engineer roles require?",
        "query": "What Python skills do AI engineer roles require?",
        "docs": [],
        "relevant": False,
        "answer": "",
        "retries": 0
    })

    print(result["answer"])
    