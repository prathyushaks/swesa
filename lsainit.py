###### LSA initial guess #########
# this script performs SVD on the word count matrices
# to learn intial guess LSA vectors. Counts can be
# raw word counts or Tf-Idf weights or even PPMI. This
# script uses raw word counts.

###### Variable description ######
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data_path = '/Users/prathyusha/codesforsethares/yelp_labelled.txt'
dobj = open(data_path,'r')
data = dobj.readlines()
for i in range(len(data)):
	data[i] = data[i].rstrip()
	data[i] = data[i].lower()
	data[i] = data[i].split()
	data[i] = ' '.join(data[i][0:-1])
# add further preprocessing as needed #


vect = CountVectorizer(min_df=1, token_pattern ='\\b\\w+\\b')
FeatrMat = vect.fit_transform(data).toarray()
Featrs = vect.get_feature_names()

num_docs = FeatrMat.shape[0]

## perform SVD to determine LSA vectors ###
[U,S,V] = np.linalg.svd(np.transpose(FeatrMat))
LSA_word = np.dot(U[:,0:len(S)],np.diag(S))
# Further reduce dimension of word embeddings as needed
dim = 50
[U,S,V] = np.linalg.svd(LSA_word)
LSA_red_word = np.dot(U[:,0:dim],np.diag(S[0:dim]))
for i in range(num_docs):
	LSA_red_word[i,:] = LSA_red_word[i,:]/(np.linalg.norm(LSA_red_word[i,:]))

### load lsa guess vectors #########
word_mat_df = pd.DataFrame(data=LSA_red_word)
word_mat_df.to_hdf('lsaguessvectors.h5','lsaguessvectors')
