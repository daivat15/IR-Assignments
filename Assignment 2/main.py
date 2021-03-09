import string
import pandas as pd
import numpy as np
import time
from nltk.tokenize import word_tokenize

from lsh_utils import shingle_hash

SHINGLE_LENGTH = 1                           # Set size of shingle
DOCUMENTS_INDEXED = 2                      # Set number of documents to be indexed + 1
no_permutation = 128                         # Number of Permutations
band_size = 8                                # Size of Bands
no_bands = int(no_permutation/band_size)     # Calculating number of Bands


# Read file
exclude = set(string.punctuation)
data = pd.read_csv('movie_plots.csv', usecols=[1, 7])
allDocsList = data.values.tolist()

##########################################################################################
# Running Time for Preprocessing: O(n) where n is sum of length of all files
# Running Time for Retrival: O(DOCUMENTS_INDEXED)
##########################################################################################

# Remove punctuation and lower case
for j in range(0, DOCUMENTS_INDEXED):
    i = allDocsList[j]
    text = i[1]
    newText = ''.join(ch for ch in text if ch not in exclude)
    newText = newText.lower()
    i[1] = newText

words = [[]] * DOCUMENTS_INDEXED
index = 0
# Creating words
for j in range(0, DOCUMENTS_INDEXED):
    i = allDocsList[j]
    text = i[1]
    tokenText = word_tokenize(text)
    words[index].append(tokenText)
    index = index + 1
# Words stored as words[index][file_index][word_index]

# Creating shingles
index = index - 1
shingle_list = [[] for i in range(index)]
shingle_map = {}
for i in range(0, index):
    for j in range(0, len(words[index][i]) - SHINGLE_LENGTH + 1):
        shingle = []
        for k in range(0, SHINGLE_LENGTH):
            shingle.append(words[index][i][j + k])
        shingle_list[i].append(shingle)
        s_hash = shingle_hash(shingle)
        if s_hash in shingle_map:
            if i not in shingle_map[s_hash]:
                shingle_map[s_hash].append(i)
        else:
            shingle_map[s_hash] = [i]
# Shingles stored as shingle_list[file_index][shingle_index]
# This is document to shingle list
# Documents stored as shingle_map[s_hash][file_index]
# This is shingle to document array. s_hash can be generated using the shingle_hash(shingle) function


# Function to create shingles from input
def shingle_text(text):
    shingle_set = []
    newText = ''.join(ch for ch in text if ch not in exclude)
    newText = newText.lower()
    tokenText = word_tokenize(newText)
    for i in range(0, len(tokenText) - SHINGLE_LENGTH + 1):
        shingle = []
        for j in range(0, SHINGLE_LENGTH):
            shingle.append(tokenText[i + j])
        shingle_set.append(shingle)
    return shingle_set


# creating shingle frequency matrix
start_time = time.time()
frequency_doc1 = []
for key, value in shingle_map.items():
    doc = []
    ik = 0
    while ik < DOCUMENTS_INDEXED:
        if ik in value:
            doc.append(1)
        else:
            doc.append(0)
        ik = ik+1
    frequency_doc1.append(doc)
print('It took %s seconds to create frequency document.' %(time.time()-start_time))
frequency_doc2 = pd.DataFrame(frequency_doc1)
frequency_doc2.index = shingle_map
# Shingles frequency matrix stored as frequency_doc1
# frequency_doc1 is a list
# This list is converted to a data frame frequency_doc2


# Function for Min hashing by creating random permutations of Shingles frequency matrix
# Selecting first row marked as 1 in each column
def min_hashing(no_p):
    start_time = time.time()
    l2 = []
    l = [t for t in range(len(frequency_doc2))]
    for t in range(no_p):
        l3 = np.random.permutation(l)
        l2.append(l3)
    l4 = []
    for j in range(no_p):
        frequency_doc3 = frequency_doc2.iloc[l2[j], :].values
        l5 = []
        for k in range(DOCUMENTS_INDEXED-1):
            l5.append(frequency_doc3[:, k].argmax())
        l4.append(l5)
    print('It took %s seconds for min_hashing.' % (time.time() - start_time))
    return l4


frequency_doc4 = min_hashing(no_permutation)
shin_matrix = pd.DataFrame(frequency_doc4)
# Shingling Matrix stored as frequency_doc4
# frequency_doc4 is a list
# This list is converted to a data frame shin_matrix


# Function for creating bands
def creating_bands():
    start_time = time.time()
    doc = shin_matrix.values
    bands_list = []
    for k in range(no_bands):
        bands_list.append(doc[k * band_size:(k + 1) * band_size, :])
    print('It took %s seconds to create bands.' % (time.time() - start_time))
    return bands_list


bands = creating_bands()
# Shingling Matrix after creating bands stored as bands


# Function for creating candidate pairs using euclidean distance as the selection
def euclid_dist():
    start_time = time.time()
    delta = 0.5
    buck_lis = [[] for i in range(no_bands)]
    for i in range(no_bands):
        np_list = np.array([[0.0 for i in range(DOCUMENTS_INDEXED-1)] for i in range(DOCUMENTS_INDEXED-1)])
        for j in range(DOCUMENTS_INDEXED-1):
            for k in range(j, DOCUMENTS_INDEXED-1):
                np_list[j][k] = ((((bands[i][:, j] - bands[i][:, k]) ** 2).sum()) ** 0.5)
                np_list[k][j] = np_list[j][k]
        np_list = np_list / np_list.max()
        for j in range(band_size):
            for k in range(j, band_size):
                if np_list[j][k] < delta:
                    buck_lis[i].append((j, k, np_list[j][k]))
    print('It took %s seconds to euclidean distance buckets .' % (time.time() - start_time))
    return buck_lis


euc_bucket = euclid_dist()          # candidate pairs stored in euc_bucket
print(euc_bucket)


# Function for creating candidate pairs using cosine distance as the selection
def cos_dist():
    start_time = time.time()
    delta = 0
    buck_lis = [[] for i in range(no_bands)]
    for i in range(no_bands):
        np_list = np.array([[0.0 for i in range(DOCUMENTS_INDEXED - 1)] for i in range(DOCUMENTS_INDEXED - 1)])
        for j in range(DOCUMENTS_INDEXED - 1):
            for k in range(j, DOCUMENTS_INDEXED - 1):
                np_list[j][k] = (((bands[i][:, j]*bands[i][:, k]).sum())/((((bands[i][:, j]**2).sum())*((bands[i][:, k]**2).sum()))**0.5))
                np_list[k][j] = np_list[j][k]
        np_list = np_list / np_list.max()
        for j in range(band_size):
            for k in range(j, band_size):
                if np_list[j][k] > delta:
                    buck_lis[i].append((j, k, np_list[j][k]))
    print('It took %s seconds to cosine distance buckets .' % (time.time() - start_time))
    return buck_lis


cos_bucket = cos_dist()         # candidate pairs stored in cos_bucket
print(cos_bucket)


# Function for creating candidate pairs using hamming distance as the selection
def ham_dist():
    start_time = time.time()
    delta = 1
    buck_lis = [[] for i in range(no_bands)]
    for i in range(no_bands):
        np_list = np.array([[0.0 for i in range(DOCUMENTS_INDEXED - 1)] for i in range(DOCUMENTS_INDEXED - 1)])
        for j in range(DOCUMENTS_INDEXED - 1):
            for k in range(j, DOCUMENTS_INDEXED - 1):
                np_list[j][k] = (((bands[i][:, j]-bands[i][:, k]) != 0.0).sum())
                np_list[k][j] = np_list[j][k]
        np_list = np_list / np_list.max()
        for j in range(band_size):
            for k in range(j, band_size):
                if np_list[j][k] < delta:
                    buck_lis[i].append((j, k, np_list[j][k]))
    print('It took %s seconds to hamming distance buckets .' % (time.time() - start_time))
    return buck_lis


ham_bucket = ham_dist()         # candidate pairs stored in ham_bucket
print(ham_bucket)
