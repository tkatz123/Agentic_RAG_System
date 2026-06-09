import pandas as pd

df = pd.read_csv('data/raw/AI_Engineer_Job_Data.csv')

print("First 5 Rows:")
print(df.head())

print(" ")

print("Rows, Columns")
print(df.shape)

print(df['job_description'].str.len().describe())