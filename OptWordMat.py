### Optimization with respect to the word matrix ###
### this step performs stochastic gradient descent with respect
### to the word matrix.

### opt_w performs stochastic gradinet descent for word matrix W
### learning rate decays as 1/sqrt(t)

## Input Variables legend ###
## num_epochs = number of epochs for SGD
## num_docs = number of documents in training data
## word_vec = matrix of word embeddings
## beta_mat = Tf-idf/word counts/word weights
## theta_ini = initial guess theta
## lam_w = regularization on word embeddings
## y = document labels
## w_neg = cost penalty of negative class
## w_pos = cost penalty of positive class
## bias = bias of logistic regression classifier

## Outputs ####
## Output from opt_w: word embedding matrix W and cost function 
## at ever iteration 
## moving average is implemented if averages of objectives over
## last few iterations are to be considered
import random
import math
from math import exp,log
import numpy as np
from numpy import linalg as la

def opt_w(num_epochs,num_docs,word_vec,beta_mat,theta_ini,alpha,lam_w,y,w_neg,w_pos,bias):
    counter=1
    token = 0
    obj_vec = [0]*(num_docs*num_epochs)
    num_word_vec = word_vec.shape[1]
    num_features = word_vec.shape[0]
    word_vec_avg = np.zeros((num_features,num_word_vec))
    for epoch in range(num_epochs):
        idx = list(range(0,num_docs))
        random.shuffle(idx)
        for i in idx:
            if y[i] == -1:
               w = w_neg
            else:
               w = w_pos
            token = token+1
            t_doc_vec = np.dot(word_vec,np.transpose(beta_mat[i,:]))
            const = w/(1+exp(y[i]*(np.dot(theta_ini,t_doc_vec)+bias)))
            gradient_w = const*(-y[i]*np.outer(np.transpose(theta_ini),beta_mat[i,:]))

       
            word_vec = word_vec - (alpha/(counter**0.5))*gradient_w
            for j in range(num_word_vec):
                word_vec[:,j] = word_vec[:,j]/(np.linalg.norm(word_vec[:,j]))
                
            if token > num_docs - round(num_docs**0.5,0):
               word_vec_avg = word_vec + word_vec_avg 


            doc_vec = np.dot(beta_mat,np.transpose(word_vec))
            objVal = np.zeros(num_docs)
            for k in range(num_docs):
                if y[k] == -1:
                    w = w_neg
                else:
                    w = w_pos
                objVal[k] =w*log(1+exp(-y[k]*(np.dot(theta_ini,doc_vec[k,:])+bias)))

            obj_vec[counter-1] = np.sum(objVal)/(num_docs) 
            counter = counter + 1
            
    word_vec = word_vec_avg/(round(num_docs**0.5,0))
    for i in range(num_word_vec):
        word_vec[:,i] = word_vec[:,i]/(np.linalg.norm(word_vec[:,i]))
    return(word_vec,obj_vec)
    
def moving_average(a, n= 5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n -1:]/n