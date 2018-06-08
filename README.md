# swesa

This repository contains all code scripts required to run the alternate minimization procedure
in the paper titled "Simple Algorithms for Sentiment Analysis on Sentiment Rich, Data Poor Domains."

Here is a brief decription of each script and the functions it performs and 
the order to use scripts.

STEP1: Run weightmat.py. 
This script calculates the weight matrix beta which is used to obatin document embeddings
from word embeddings. 
User can change parameters such as minimum word count, min length of word token etc 
in this script to obtain the final weight matrix and the final count of documents.

STEP2: Run initialization script lsainit.py or retrain word2vec to obtain initial guess
embeddings. Note: if running lsainit.py make sure to keep the same vocab as in STEP1.

STEP3: Run swesascript.py to perform alternate minimization to learn word embeddings
and classifier. Note that this script makes use of SplitData.py. SplitData.py returns
stratified k-fold splits of train and test data. This script can be modified as per
the requirements to get validation sets too. The script OptWordMat.py performs SGD
with respect to the word matrix as described in the paper.

The folder 'data' contains the Yelp, Amazon and IMDB data sets downloaded from
"Dimitrios Kotzias, Misha Denil, Nando De Freitas, and Padhraic Smyth. 2015. From group to individual labels using deep features. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 597–606. ACM". However, due to the proprietary nature of the A-CHESS data set it is not made publicly available. Please write to kameswarasar@wisc.edu, if you wish to access this data set.

If using this code, please cite our paper as mentioned above.

If you encounter any errors/difficulty running any of these scripts, please write to kameswarasar@wisc.edu
