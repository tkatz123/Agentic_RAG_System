from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()


def rag(question):

    #Loading database object
    db = Chroma(persist_directory = "data/chroma_db", embedding_function = OpenAIEmbeddings())

    #Retriving top 10 relevant chunks
    docs = db.similarity_search(question, k = 10)

    #Combining chunks into single string
    context = "\n\n".join([doc.page_content for doc in docs])

    #Creates stuffing prompt to feed to LLM
    input_string = f"""You are a helpful assistant. Use the context below to answer the question. If the answer is not in the context, say so

    Context:
    {context}

    Question: {question}
    Answer:

    """

    #Initialize OpenAI object
    client = OpenAI()

    #Sends query to LLM
    response = client.responses.create(
        model = "gpt-4o-mini", 
        input = input_string
        )

    print(" ")
    print("Response")
    print("=========")
    print(" ")
    print(response.output_text)

if __name__ == "__main__":
    
    rag("What kind of python skills do companies want an AI engineer to know?")





