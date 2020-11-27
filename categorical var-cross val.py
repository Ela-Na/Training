# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:41:36 2020

@author: Ela
"""

import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../input/cat_train.csv")

# create newcollumn and fill it with -1

df["kfold"] = -1

#randomize the rows of data.

df = f.sample(frac=1).reset_index(drop=True)

df.ord_2.fillna("None").value_counts()

df.ord_4.fillna("None").value_counts()

df.loc[df["ord_4"].value_counts()[df["ord_4"]].values < 2000, "ord_4"] = "RARE"


# fetch target values
y = df.target.values

# initiate kfold class from model selection module
kf = model_selection.StratifiedKFold(n_splits=5)
 
 # fill the new kfold column
 
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f
     
# save the new csv with kfold column
     
df.to_csv("../input/cat_train_folds.csv", index=False)