from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import build_graph
from openai import OpenAI


load_dotenv()


def eval_retrieval():
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

#Runs RAG pipeline with agent
def run_agent(question):
    agent = build_graph()
    result = agent.invoke({
        "question": question,
        "query": question,
        "docs": [],
        "relevant": False,
        "answer": ""
    })

    return result['answer'], result['docs']

#Runs RAG pipeline without agent
def run_simple(question):

    #Loading database object
    db = Chroma(persist_directory = "data/chroma_db", embedding_function = OpenAIEmbeddings())

    #Retriving top 10 relevant chunks
    docs = db.similarity_search(question, k = 10)

    #Combining chunks into single string
    context = "\n\n".join([doc.page_content for doc in docs])

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
    
    return response.output_text, docs

def judge_faithfullness(question, answer, context) -> int:
    client = OpenAI()
    prompt = f"""
    You are evaluating whether an answer to the question is faithful to the provided context.

    Context: {context}
    Question: {question}
    Answer: {answer}

    Sccore the answer on a scale of 1-3:
    1 = answer contradicts or ignors the context
    2 = answer is partially supported by the context
    3 = answer is fully grounded in the context

    Return only the number 1, 2, or 3. Nothing else.
    """

    #Tries three times to get the correct output
    for _ in range(3):
        response = client.responses.create(
             model="gpt-4o-mini",
            input=prompt
          )
        try:
            output_num = int(response.output_text.strip())
            if output_num in [1, 2, 3]:
                return output_num
        except ValueError:
            continue

    return 2



if __name__ == "__main__":

    df = pd.read_csv('eval/gold_set.csv')

    simple_score = 0
    agent_score = 0

    for _, row in df.iterrows():
        answer, docs = run_simple(row['question'])
        context = "\n\n".join([doc.page_content for doc in docs])
        simple_score += judge_faithfullness(row['question'], answer, context)
    
    simple_avg_score = round(simple_score / df.shape[0], 2)

    for _, row in df.iterrows():
        answer, docs = run_agent(row['question'])
        context = "\n\n".join([doc.page_content for doc in docs])
        agent_score += judge_faithfullness(row['question'], answer, context)
    
    agent_avg_score = round(agent_score / df.shape[0], 2)

    print(f"Simple Score: {simple_avg_score}")
    print(f"Agent Score: {agent_avg_score}")



