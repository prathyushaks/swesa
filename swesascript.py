# This script performs the alernate minimization objective to learn
# word embeddings and a classifier for a small sized labled data set.
# First, load the data file and split the review text from the label.
# Next load the precomputed weight matrix.
# Initialization can be done using LSA word vectors or retrained word2vec embeddings.
# Set the training parameters.
# Solve first for classifier theta and next for the word embeddings.
# Classifier can be solved using the pre-existing classifier built in Python.
# call opt_w from OptWordMat that derives updates with respect to the word matrix and 
# perform Stochastic Gradient Descent.
# Script returns classifier and word matrix at end of alternate minimization.

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, confusion_matrix,f1_score
from sklearn import metrics
import pandas as pd
from SplitData import splitData
from OptWordMat import opt_w
import matplotlib.pyplot as plt
import math
from math import log,exp

#### load data from path and split label from review #####
data_path = 'path_to_data'
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

labels = np.asarray(labels)
path_to_weight_matrix = 'path to weight matrix'
fname = 'weightmatrix.h5'
weight_matrix_df = pd.read_hdf(fname)
weight_matrix = weight_matrix_df.as_matrix()

splits = 10
trainb_ilst,trainy_ilst,testb_ilst,testy_ilst = splitData(weight_matrix,labels,splits)

split_idx = 0
#### determine train and test data #####
train_mat = np.array(trainb_ilst[split_idx])
train_labels = np.array(trainy_ilst[split_idx])

test_mat = np.array(testb_ilst[split_idx])
test_labels = np.array(testy_ilst[split_idx])

num_features = 80
#### set training parameters ##########
# load guess matrix either with LSA or word2vec
fname = 'lsaguessvectors.h5'
word_guess_df = pd.read_hdf(fname)
word_guess = word_guess_df.as_matrix()
word_guess = np.transpose(word_guess[0:vocab,:])

train_data = np.dot(train_mat,np.transpose(word_guess))
test_data = np.dot(test_mat,np.transpose(word_guess))

num_training_docs = train_data.shape[0]
lam_t=0.5
lam_w = math.pow(10,-6)
num_epochs = 2

## for balanced data sets w_neg = w_pos ###
## change weights for imbalanced data sets
w_neg=1
w_pos=1

w_par = dict([(-1,w_neg),(1,w_pos)])

alpha = 0.008
alpha2 = 0.4
tol = math.pow(10,-5)
tol1 = math.pow(10,-4)
token = 9

num_iter = 15
step = 1
obj_val_vec =[0]*(num_iter+1)

################ Alternate minimization on training data ########
for iters in range(num_iter):
      
    classifier = SGDClassifier(loss='log',alpha=alpha,eta0=0.0,learning_rate='optimal',
                               penalty='l2',class_weight=w_par,average=True)
    clsf = classifier.fit(train_data,train_labels)
    theta_est = clsf.coef_
    bias =clsf.intercept_

    [word_vec,o2]  = opt_w(num_epochs,num_training_docs,word_guess,train_mat,
                              theta_est,alpha2,lam_w,train_labels,w_neg,w_pos,bias) 
                              
    doc_vec_tr = np.dot(train_mat,np.transpose(word_vec))
    word_guess = word_vec
    
    ## calculate total bi-convex objective and monitor###  
    objVal = np.zeros(num_training_docs)
    for j in range(num_training_docs):
        if train_labels[j] == -1:
            w = w_neg
        else:
            w = w_pos
        objVal[j] = w*log(1+exp(-train_labels[j]*(np.dot(theta_est,doc_vec_tr[j,:])+bias)))
        
    obj_val_vec[step] =np.sum(objVal)/(num_training_docs) + lam_w* np.linalg.norm(
            word_vec,'fro')**2 + alpha*lam_t*np.linalg.norm(theta_est)**2;
            

    step = step+1
    
############# plot objective function ######
plt.figure(1)    
plt.plot(obj_val_vec[1:iters],'-or')
plt.show()
plt.xlabel('iterations')
plt.ylabel('Objective function')
plt.title('Alternate minimization of objective')

###### save word vectors as h5 file ########

pol_wordmat_df = pd.DataFrame(data=word_vec)
pol_wordmat_df.to_hdf('polarized_wordvectors.h5','polarized_wordvectors')