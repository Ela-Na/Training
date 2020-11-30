# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:04:13 2020

@author: Ela
"""

import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

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
    
    # initiate logistic regression model
    
    model = linear_model.LogisticRegression()
    
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
        
        
