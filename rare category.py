# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:45:39 2020

@author: Ela
"""

import pandas as pd
from sklearn import preprocessing

train = pd.read_cv("../input/cat_train.csv")

test = pd.read_csv("../input/cat_test.csv")

# create a fake target column for test data, since this column doen't exist

test.loc[:, "target"] = -1

# concatenating both training and test data 

data = pd.concat([train, test]).reset_index(drop=True)

# make a list of features we are interested in id and target are not encode

features = [x for x in train.columns if x not in ["id", "target"]]


# loop over the feature list

for f in features:
    
    # create a new instance of LabelEncoder for each feature
    lbl_enc = preprocessing.LabelEcoder()
    
# since it is categorical data, fill the nan values with a string
# convert all data to string
    
    temp_col = data[f].fillna("None").astype(str).values
    
    # can use fit_transform 
    data.loc[:, f] = lbl_enc.fit_transform(temp_col)
    
    # split train and test data again
    
    train = data[data.target != -1].reset_index(drop=True)
    test = data[data.target == -1].reset_index(drop=True)
    
    