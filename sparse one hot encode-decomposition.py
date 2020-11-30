# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:31:10 2020

@author: Ela
"""

import pandas as pd

from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    
    #load the full training data with folds
    
    df = pd.read_csv("../input/cat_train_folds.csv")
    
    # all columns except target, id and kfold are features
    
    features = [ f for f in df.columns if f not in ("id", "target" , "kfold")]
    
    # covert all columns to string and fill all Nan values with None
    
    for col in features:
        
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
     # train data using folds
        
    df_train = df[df.kfold != fold].reset_index(drop=True)
        
        # validate data using folds
        
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # use onehotencoder fron scikit_learn
        
    ohe = preprocessing.OneHotEncoder()
    
    # fit ohe on training and validation features
        
    full_data = pd.concat(
                [df_train[features], df_valid[features]], axis=0)
    
    ohe.fit(full_data[features])
        
        # transform training data and validation data
        
    x_train = ohe.transform(df_train[features])
        
    x_valid = ohe.transform(df_valid[features])
    
    # initialize Truncated SVD
    
    # reducing data to 120 components
    
    svd = decomposition.TruncatedSVD(n_components=120)
    
    # fit svd on full sparse training data
    
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)
    
    # transform sparse training and validation data
    
    x_train = svd.transform(x_train)
    
    x_valid = svd.transform(x_valid)
    
     # initiate lrandom forest model
    
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    
    # fit model on training data (ohe)
    
    model.fit(x_train, df_train.target.values)
    
     # predict on validation data , need probability values for calculating AUC
    
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get roc auc score
    
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    #print auc
    
    print(auc)
    
if __name__ == "__main__" :
        
        # run function for fold 0, we can replace 0 and run it with any fold number
   for fold_ in range(5):
      run(fold_)
    
    
