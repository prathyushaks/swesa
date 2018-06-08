### this script calculates word weights/word counts of words within documents
### word frequency is set in min_count. Words appearing less than min_count
### will not be part of the final vocabulary. Additional word tokens that
### are greater than 2 in length are considered.

##### Variables Key #######
#min_count determines the min number of times a word has to appear in the corpus.
#red_vocab = pruned vocabulary based on min word frequency.
#beta_mat  = matrix of word weights. 
#rel_docs = documents containing all words in vocabulary and non empty documents.
#corpus = final reviews/text satisfying min word count requirements.


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data_path = '/Users/prathyusha/codesforsethares/yelp_labelled.txt'
dobj = open(data_path,'r')
data = dobj.readlines()
reviews = list(range(len(data)))
labels = list(range(len(data)))
for i in range(len(data)):
	data[i] = data[i].lower()
	data[i] = data[i].rstrip()
	data[i] = data[i].split()
	reviews[i] = ' '.join(data[i][0:-1])
	labels[i] = int(data[i][-1])
##### convert labels with 0 to -1 ###########
for i in range(len(labels)):
	if labels[i]==0:
		labels[i]=-1

num_docs = len(reviews)
vect = CountVectorizer(min_df=1, token_pattern ='\\b\\w+\\b')
temp_FeatrMat = vect.fit_transform(reviews).toarray()
temp_Featrs = vect.get_feature_names()

test = np.sum(temp_FeatrMat,axis=0)
min_count =2

red_vocab=[]
red_vocab_idx=[]
for i in list(range(len(test))):
    idx = test[i]
    if idx >= min_count:
        word = temp_Featrs[i]
        if len(word) >2: 
            red_vocab.append(word)
            red_vocab_idx.append(i)
        
red_vocab = red_vocab[2:]
red_vocab_idx = red_vocab_idx[2:]
red_count_mat = np.zeros((num_docs,len(red_vocab)))
for i in list(range(len(red_vocab_idx))):
    idx = red_vocab_idx[i]
    red_count_mat[:,i] = temp_FeatrMat[:,idx]
    
beta_mat = red_count_mat ## weights
rel_docs = np.sum(beta_mat,axis=1)
ret_idx = []

for i in list(range(len(rel_docs))):
    relv = rel_docs[i]
    if relv >0:
        ret_idx.append(i)

beta_mat = beta_mat[ret_idx,:]
for i in range(beta_mat.shape[0]):
    beta_mat[i,:] = beta_mat[i,:]/np.sum(beta_mat[i,:])


####### return list of reviews and labels that satisfy min count requirements ######
labels = [labels[i] for i in ret_idx]
corpus = list(range(len(labels)))
for i in range(len(corpus)):
    corpus[i] = reviews[ret_idx[i]]+' '+str(labels[i])
    corpus[i] = ''.join(corpus[i])

######### save weight matrix and write reviews and labels as text files #####
wobj = open('final_data.txt','w')
for i in corpus:
	wobj.write('%s\n'%i)

weight_mat_df = pd.DataFrame(data=beta_mat)
weight_mat_df.to_hdf('weightmatrix.h5','weightmatrix')