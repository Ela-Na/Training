# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:49:45 2020

@author: Ela
"""

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    
    # read dataset with folds 
    
    df = pd.read_csv("../input/mnist_train_folds.csv")
    
    # training data where kfold is not equal to provided fold, reset the index
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
        # validation data where kfold is  equal to provided fold, reset the index

    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # drop label column in dataframe and convert it to numpy array using values- 
    # target is label column in dataframe
    
    x_train = df_train.drop("label", axis = 1).values
    y_train = df_train.label.values
    
    x_valid = df_valid.drop("label", axis = 1).values
    y_valid = df_valid.label.values
    
    
    
    # iniytialize decision tree classifier
    
    clf = tree.DecisioTreeClassifier()
    
    # fit model on traning data
    
    clf.fit(x_trian, y_train)
    
    # create predictions on validation samples
    
    preds = clf.predict(x_valid)
    
    # calculate and print accuracy
    
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold=(fold), Accuracy={accuracy}")
    
    
    # save the model
    
    joblib.dump(clf, f"../models/dt_{fold}.bin")
    
    
if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)
    
    
