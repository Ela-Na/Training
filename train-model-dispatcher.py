# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:17:08 2020

@author: Ela
"""

import os
import config
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import model_dispatcher

def run(fold):
    
    # read dataset with folds 
    
    df = pd.read_csv(config.training_file)
    
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
    
    clf = model_dispatcher.models[model]
    
    # fit model on traning data
    
    clf.fit(x_trian, y_train)
    
    # create predictions on validation samples
    
    preds = clf.predict(x_valid)
    
    # calculate and print accuracy
    
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold=(fold), Accuracy={accuracy}")
    
    
    # save the model
    
    joblib.dump(clf, os.path.join(config.model_output, f"dt_{fold}.bin))
    
    
if __name__ == "__main__":
    
    #initialize argumentparser class for argparse
    
    parser = argparse.ArgumentParser()
    
    # add different arguments we need and ther types
    # we need fold
    
    parser.add_argument(
            "--fold",
            type=int
            )
    
    
     parser.add_argument(
            "--model",
            type=int
            )
    # read arguments from command line
    
    args = parser.parse_args()
    
    # run the fold specified by command line arguments
    
    run(fold=args.fold, model=args.model)