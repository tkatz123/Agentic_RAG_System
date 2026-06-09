import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()


#Reading in job data
df = pd.read_csv('data/raw/AI_Engineer_Job_Data.csv')

#Initialize splitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

chunk_elements = []

#Loop over dataframe and chunk each job description. Attach associated meta data via dictionary
for row, col in df.iterrows():
    chunks = splitter.create_documents([col['job_description']], [{'company': col['company'], 'job_title': col['job_title']}])
    chunk_elements.extend(chunks)

print(f"Total Chunk: {len(chunk_elements)}")
print(chunk_elements[0])

#Define embedding object
embedding_obj = OpenAIEmbeddings()

db = Chroma.from_documents(chunk_elements, embedding_obj, persist_directory="data/chroma_db")
print(f"Documents in Chroma: {db._collection.count()}")