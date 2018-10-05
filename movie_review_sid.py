import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Init Objects
sw = set(stopwords.words('english'))
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def getCleanReview(str):
    
    str = str.lower()
    str = str.replace('<br /><br />', '')
    
    # Tokenize
    tokens = tokenizer.tokenize(str)
    new_tokens = [token for token in tokens if token not in sw]
    stemmed_tokens = [ps.stem(x) for x in new_tokens]
    cleaned_review = ' '.join(stemmed_tokens)
    return cleaned_review

# Function that takes input file and return cleaned output file of movie review
def getCleanDocument(inputFile,outputFile):

	out = open(outputFile,'w',encoding="utf8")

	with open(inputFile,encoding="utf8") as f:
		reviews = f.readlines()

	for review in reviews:
		cleaned_review = getCleanReview(review)
		print((cleaned_review),file=out)

	out.close()
    

# Read Command Line Arguments
inputFile = sys.argv[1]
outputFile = sys.argv[2]
getCleanDocument(inputFile, outputFile)
    
    
    
    
    
    
    