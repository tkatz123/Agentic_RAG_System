import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()


#Reading in data
job_df = pd.read_csv('data/raw/AI_Engineer_Job_Data.csv')
arxiv_df = pd.read_csv('data/raw/arxiv_abstracts.csv')

#Initialize splitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

job_chunks = []
abstract_chunks = []

#Loop over dataframe and chunk each job description. Attach associated meta data via dictionary
for row, col in job_df.iterrows():
    chunks = splitter.create_documents([col['job_description']], [{'company': col['company'], 'job_title': col['job_title'], 'source': 'job_posting'}])
    job_chunks.extend(chunks)

print(f"Total Job Chunks: {len(job_chunks)}")
print(job_chunks[0])

#Loop over dataframe and chunk each paper description. Attach associated meta data via dictionary
for row, col in arxiv_df.iterrows():
    chunks = splitter.create_documents([col['abstract']], [{'title': col['title'], 'source': 'arxiv'}])
    abstract_chunks.extend(chunks)

print(f"Total Abstract Chunks: {len(abstract_chunks)}")
print(abstract_chunks[0])

all_chunks = job_chunks + abstract_chunks

#Define embedding object
embedding_obj = OpenAIEmbeddings()

db = Chroma.from_documents(all_chunks, embedding_obj, persist_directory="data/chroma_db")
print(f"Documents in Chroma: {db._collection.count()}")