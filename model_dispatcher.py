# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:11:41 2020

@author: Ela
"""

from sklearn import tree

models = {
        "decision_tree_gini": tree.DecisionTreeClassifier(
                criterion="gini"),'
                "decision_tree_entropy":
                    tree.DecisionTreeClassifier(criterion="entropy"),
                    "rf": ensemble.RandomForestClassifier(),
                    }