import pandas as pd
import arxiv

#Create list of queries to search for articles
search_queries = ['retrieval augmented generation', 'LLM agents', 'large language models', 'neural networks', 'containerization']

#Initialize client object
client = arxiv.Client()

papers = []

#Iterate through queries, storing article title and abstract in a list
for query in search_queries:
    search = arxiv.Search(query=query, max_results=50)
    for paper in client.results(search):
        papers.append({'title': paper.title, 'abstract': paper.summary})

#Convert list to dataframe
df = pd.DataFrame(papers)

#Confirm dataframe creation
print(df.head())
print(len(df))

#Exporting dataframe as a CSV
df.to_csv('data/raw/arxiv_abstracts.csv')

