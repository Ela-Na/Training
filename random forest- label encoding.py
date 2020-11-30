# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 21:24:20 2020

@author: Ela
"""

import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    df = pd.read_csv("../input/cat_train_folds.csv")
    
# all columns except target, id and kfold are features
    
    features = [ f for f in df.columns if f not in ("id", "target" , "kfold")]
    
    # covert all columns to string and fill all Nan values with None
    
    for col in features:
        
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
     # label encode the features   
    for col in features:
        
        lbl = preprocessing.LabelEncoder()
        
        # fit label encoder on all data
        
        lbl.fit(df[col])
        
        # transform all the data
        
        df.loc[:, col] = lbl.transform(df[col])
        
        # train data using folds
        
    df_train = df[df.kfold != fold].reset_index(drop=True)
        
        # validate data using folds
        
    df_valid = df[df.kfold == fold].reset_index(drop=True)
                
    # training data
    
    x_train = df_train[features].values
    
    # get validation data
    
    x_valid = df_valid[features].values
    
    # initiate logistic regression model
    
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
        
        