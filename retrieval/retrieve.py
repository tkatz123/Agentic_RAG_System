from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

db = Chroma(persist_directory = "data/chroma_db", embedding_function = OpenAIEmbeddings())

test_query = "What Python skills do AI enginenr roles require?"

query_result = db.similarity_search(test_query, k = 5)

for doc in query_result:
    print(doc.page_content)
    print(doc.metadata)
    print("----------------------")
    print(" ")