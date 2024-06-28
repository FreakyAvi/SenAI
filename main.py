import pandas as pd
import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
import torch
import re

def SApipeline(text):
    #creating tokens
    tokens = tokenizer.encode(text, return_tensors='pt')
    #print(tokens)       # check the tokens

    #pass tokens to model
    results = model(tokens)
    #print(results)  #check results

    #print(f"Logits: {results.logits}") # Check logits

    SAscore = int(torch.argmax(results.logits)) + 1
    #print(f"Sentiment Score:{SAscore}")
    return(SAscore)

#initializing model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
t = "it is okay"
# SApipeline(t)

df = pd.read_csv("AmazonProdReviews.csv")
# print(df.head())

if "SA Score" not in df.columns.tolist():
    df["SA Score"] = None

for i in range(len(df)):
    if df.loc[i, "SA Score"] not in [1,2,3,4,5]:
        df.loc[i, "SA Score"] = SApipeline(df.loc[i, "reviews"])

df.to_csv("AmazonProdReviews.csv", index=False)
print("PROCESS COMPLETED!")
print(df.tail())





