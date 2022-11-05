#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Facundo
"""

from BuildDecisionTree_1 import BuildDecisionTreeClass


tree = BuildDecisionTreeClass(file = "file1.txt", minNum = 2)
data = tree.getTable()
alpha = 0.5

def trainError(tree, data):
    root = tree.getRoot()
    totalError = treePath(root)
    return totalError

def treePath(node):
    Error = 0
    
    if node.t == "Leaf":
        Error += node.cv[1-node.v]
    else:
        for i in range(len(node.c)) :
            i=i+1
            if (i == 1):
                Error += treePath(node.l)
            if (i == 2):
                Error += treePath(node.r)
    return Error
    
    
def complexity(tree):
    result = tree.Print()
    complexity = result.count("Leaf")
    return complexity 

def generalizationError(data, tree, alpha = 0.5):
    genError = ( trainError(tree, data) + alpha*complexity(tree) )
    return genError
    
"""
Function to execute
"""
#print(generalizationError(data,tree, alpha))