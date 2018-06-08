# This script splits input data into stratified splits of
# training and test data sets.
# Stratified split is recommended inorder to maintain a 
# uniform distribution of minor and major classes in train
# and test data sets.

from sklearn.model_selection import StratifiedKFold

def splitData(beta_mat,labels,splits):
    skf = StratifiedKFold(n_splits=splits)
    skf.split(beta_mat,labels)
    trainb_lst = []
    testb_lst = []

    trainy_lst=[]
    testy_lst=[]
    for train_index, test_index in skf.split(beta_mat,labels):
        trainb_lst.append(beta_mat[train_index])
        testb_lst.append(beta_mat[test_index])
        trainy_lst.append(labels[train_index])
        testy_lst.append(labels[test_index])
        
    return(trainb_lst,trainy_lst,testb_lst,testy_lst)

    
    
