import pandas as pd
import string
import math

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

exclude = set(string.punctuation)
data = pd.read_csv('movie_plots.csv', usecols=[1, 7])
allDocsList = data.values.tolist() #List of all documents with 1st column as movie name and 2nd one as movie plot
n_ds = 200
print("Removing punctuation")
for i in allDocsList[0:n_ds]:
    #print(i)
    text = i[1]
    newText = ''.join(ch for ch in text if ch not in exclude)
    i[1] = newText

print("Punctuation removed")

dict = {}

for i in allDocsList[0:n_ds]:
    text = i[1]
    tokenText = word_tokenize(text) #text tokenization
    lowers = [word.lower() for word in tokenText]
    i[1] = lowers #lower case
    
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

for i in allDocsList[0:n_ds]:      #stemming and removal of stop words
    filtered_sentence = [w for w in i[1] if not w in stop_words]
    stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
    i[1] = stemmed_sentence

for i in range(0,n_ds):            #making inverted index
    for key in allDocsList[i][1]:
        if key in dict.keys():
            dict[key].append(i)
        else:
            temp = []
            temp.append(i)
            dict[key] = temp

def tf(word, doc):
    termF = 0
    if word in dict.keys():
        for i in dict[word]:
            if i==doc:
                termF += 1
    if termF == 0:
        return 0
    else:
        return 1 + math.log(termF)
    
def idf(word):
    if word in dict.keys():
        docs = dict[word]
        docF = set(docs)
        #print(docF)
        return math.log(n_ds/len(docF))
    else:
        return 0

def searchQuery(query):
    newText = ''.join(ch for ch in query if ch not in exclude)
    tokenText = word_tokenize(newText) 
    lowers = [word.lower() for word in tokenText]
    filtered_sentence = [w for w in lowers if not w in stop_words]
    stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
    query = stemmed_sentence
    docs = []
    for w in query:
        if w in dict.keys():
            for i in dict[w]:
                docs.append(i)
    docs = list(set(docs))
    q_len = len(query)
    d_len = len(docs)
    q_values = {}
    d_values = [[0 for i in range(d_len)] for j in range(q_len)]
    idf_values = {}
    for w in query:
        idf_values[w] = idf(w)
    #print(idf_values)
    for w in query:
        q_values[w] = 0
    for w in query:
        q_values[w] += 1
    for w in query:
        if q_values[w] != 0:
            q_values[w] = 1 + math.log(q_values[w])
        q_values[w] *= idf_values[w]
    
    for w in range(0, q_len):
        for d in range(0, d_len):
            word = query[w]
            document = docs[d]
            d_values[w][d] = tf(word,document)
    #print(d_values)
    for d in range(0, d_len):
        wt = 0
        for w in range(0, q_len):
            wt += d_values[w][d]*d_values[w][d]
        #print(wt)
        if wt != 0:
            for w in range(0, q_len):
                d_values[w][d] = (d_values[w][d]/math.sqrt(wt))*idf_values[query[w]]
    #print(d_values)
    wt = 0
    for w in range(0, q_len):
        wt += q_values[query[w]]*q_values[query[w]]
    if wt != 0:
        for w in range(0, q_len):
            q_values[query[w]] /= math.sqrt(wt)
    cos_values = []
    for d in range(0, d_len):
        val = 0
        for w in range(0, q_len):
            val += q_values[query[w]]*d_values[w][d]
        cos_values.append(val)
    #print(cos_values)
    print(query)
    found = 0
    count = 0
    for d in range(0, d_len):
        max = 0
        for t in range(0, d_len):
            if cos_values[t] > cos_values[max]:
                max = t
        found = 1
        print(allDocsList[docs[max]][0])
        cos_values[max] = 0
        count = count + 1
        if count == 10:
            break
    if found == 0:
        print("No movie found")
    return

searchQuery("Chaplin plays a waiter who fakes being a Greek Ambassador to impress a girl He then is invited to a garden party where he gets in trouble with the")