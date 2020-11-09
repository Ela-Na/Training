# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:44:50 2020

@author: Ela
"""

import argparse

if __name__ == "__main__":
    
    #initialize argumentparser class for argparse
    
    parser = argparse.ArgumentParser()
    
    # add different arguments we need and ther types
    # we need fold
    
    parser.add_argument(
            "--fold",
            type=int
            )
    # read arguments from command line
    
    args = parser.parse_args()
    
    # run the fold specified by command line arguments
    
    run(fold=args.fold)