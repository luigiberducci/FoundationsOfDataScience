# Author: Luigi Berducci
# Data: 06 Feb 2019
# Purpose: implement ID3 algorithm

from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import pandas as pd
import math
import datetime

class ID3:
    def __init__(self, data, dependentVariable):
        self.tree = Node("Root")
        self.data = data
        self.var  = dependentVariable
        self.values = set( data[self.var].values )

    def createDecisionTree(self):
        attributes = self.data.columns.to_list()
        attributes.remove(self.var)
        self.recursiveTree(self.data, attributes, self.tree)

    def recursiveTree(self, data, attributes, parentNode):
        if(len(attributes)==0):
            return
        bestAttribute = self.getMaxGainNode(data, attributes)
        for value in set(data[bestAttribute].values):
            newData = data[ data[bestAttribute] == value ]
            newAttributes = attributes.copy()
            newAttributes.remove(bestAttribute)
            if(self.isCompletelyInformative(newData)):
                result = newData[self.var].unique()[0]
                currentNode = Node("{}=={} -> {}".format(bestAttribute, value, result), parentNode)
            else:
                currentNode = Node("{}=={}".format(bestAttribute, value), parentNode)
                self.recursiveTree(newData, newAttributes, currentNode)

    def isCompletelyInformative(self, data):
        outcome = data[self.var]
        return len(outcome.unique())==1

    def getMaxGainNode(self, data, attributes):
        maxGain = 0
        root = None
        for att in attributes:
            attGain = self.computeInformationGain(data, att)
            if(attGain>=maxGain):
                root = att
                maxGain = attGain
        return root

    def computeInformationGain(self, data, attribute):
        infoGain = self.computeEntropy(data)
        nInstances = len(data)
        attributeValues = set( data[attribute].values )
        for val in attributeValues:
            subset = data[ data[attribute]==val ]
            nSubset = len(subset)
            valEntropy = self.computeEntropy(subset)
            infoGain += - (nSubset/nInstances) * valEntropy
        return infoGain

    def computeEntropy(self, data):
        entropy = 0
        nInstaces = len(data)
        for val in self.values:
            subset = data[ data[self.var]==val ]
            nSubset = len(subset)
            ratio = nSubset/nInstaces
            if(ratio==0):
                entropy += 0
            else:
                entropy += -ratio * math.log2(ratio)
        return entropy

    def printTree(self):
        if(self.tree==None):
            return
        for pre, fill, node in RenderTree(self.tree):
            print("{}{}".format(pre, node.name))
