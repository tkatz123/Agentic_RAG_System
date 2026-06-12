from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

#Read in data
gold_set = pd.read_csv('eval/gold_set.csv')

#Initialize database object
db = Chroma(persist_directory = "data/chroma_db", embedding_function = OpenAIEmbeddings())



for k in [3, 5, 10]:

    #Set initial score to 0
    score = 0
    
    #Iterate over the dataframe
    #Compares expected result to actual result, adds 1 to score if correct
    for _, row in gold_set.iterrows():
        question = row['question']
        expected_source = row['expected_source']

        #Searches for similar chunks, determined by k
        docs = db.similarity_search(question, k = k)

        #Uses a different source depending if it's a job posting or arxiv document
        for doc in docs:
            if doc.metadata['source'] == 'job_posting':
                retrieved_source = doc.metadata['company']
            else:
                retrieved_source = doc.metadata['title']
            
            if retrieved_source == expected_source:
                score += 1
                break

    #Divides score by 20 (total amount of expected Q&A)
    eval_score = (score/20) * 100

    print(f"Hit Rate: {eval_score}% with K = {k}")

