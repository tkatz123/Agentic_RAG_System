from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()


gold_set = pd.read_csv('eval/gold_set.csv')

db = Chroma(persist_directory = "data/chroma_db", embedding_function = OpenAIEmbeddings())

score = 0

for _, row in gold_set.iterrows():
    question = row['question']
    expected_source = row['expected_source']

    docs = db.similarity_search(question, k = 5)

    for doc in docs:
        if doc.metadata['source'] == 'job_posting':
            retrieved_source = doc.metadata['company']
        else:
            retrieved_source = doc.metadata['title']
        
        if retrieved_source == expected_source:
            score += 1
            break

eval_score = (score/20) * 100

print(f"Hit Rate: {eval_score}%")

