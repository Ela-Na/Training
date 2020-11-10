# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:44:31 2020

@author: Ela
"""

import pandas as pd
from sklearn import preprocssing

df = pd.read_csv("../imput/cat_train.csv")

# fill Nan values in ord_2 column

df.loc[:, "ord_2"] = df.ord_2.fillna("NONE")

# initialize labelencoder

lb_ec = preprocessing.LabelEncoder()

# fit label encodr and transform values on ord_2 column
# P.S: do not use this directly. ft first, then transform

df.loc[:, "ord_2"] = lb_ec.fit_transform(df.ord_2.values)