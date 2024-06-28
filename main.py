import pandas
import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
import torch
import re

#initializing model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#creating tokens
tokens = tokenizer.encode('THIS IS THE BEST THING THAT HAS ECER HAPPENED TO ME THANK YOU SO MUCH!', return_tensors='pt')
print(tokens)       # check the tokens

#pass tokens to model
results = model(tokens)
print(results)  #check results

print(f"Logits: {results.logits}") # Check logits

SAscore = int(torch.argmax(results.logits)) + 1
print(f"Sentiment Score:{SAscore}")
