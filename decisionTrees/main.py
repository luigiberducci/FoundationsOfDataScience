# Author:   Luigi Berducci
# Date:     06 Feb 2019
# Purpose:  implement ID3 algorithm to create a decision tree which fit a boolean-function over the training data

import os
import sys
import argparse
import pandas as pd
import numpy as np
import ID3

def main(argv=[]):
    parser = argparse.ArgumentParser("ID3 Implementation")
    parser.add_argument('--train', help="Path to the training dataset")
    parser.add_argument('--test', help="Path to the test dataset")
    args = parser.parse_args()

    train = args.train
    test  = args.test
    if(args.train==None):
        train = os.path.join("data", "tennis.csv")
    if(args.test==None):
        test = os.path.join("data", "unseen_tennis.csv")

    trainSet, decisionVar = loadInputDataset(train)
    testSet = loadTestSet(test)

    for row in testSet.iterrows():
        pred = ID3.predict(row)
        print(pred)
        #TODO

    id3 = ID3.ID3(trainSet, decisionVar)
    id3.createDecisionTree()

    id3.printTree()

def loadInputDataset(path):
    dataset = pd.read_csv(path)
    attributes = dataset.columns
    return dataset, attributes[-1]

def loadTestSet(path):
    test = pd.read_csv(path)
    return test

if __name__=="__main__":
    main(sys.argv)
