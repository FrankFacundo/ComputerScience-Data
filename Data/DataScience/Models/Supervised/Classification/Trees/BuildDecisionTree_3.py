#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: frank
"""

from BuildDecisionTree_1 import BuildDecisionTreeClass
from BuildDecisionTree_2 import generalizationError

tree = BuildDecisionTreeClass(file="file1.txt", minNum=2)
alpha = 0.5
minNumP = 2

import copy


def Postpruning(tree, alpha, minNumP):
    original_tree = copy.deepcopy(tree)
    final_tree = copy.deepcopy(tree)
    listNodesbyLevels = listNodesinLevels(tree)
    data = tree.getTable()
    gen_error_T = generalizationError(data, original_tree, alpha)

    for i in range(MaxLevel(tree) - 1, 0, -1):
        for node in listNodesbyLevels[i - 1]:
            tree = original_tree
            reduceTree(node.id, tree)
            gen_error_P = generalizationError(data, tree, alpha)
            if gen_error_P < gen_error_T:
                final_tree = copy.deepcopy(tree)

    print(final_tree.Print())
    return


def MaxLevel(tree):
    result = tree.getResult()
    numberOfLeaves = result.count("Leaf")
    levels = result.split("Leaf\nLevel ", numberOfLeaves)
    levelMax = 0

    for i in range(len(levels) - 1):
        i += 1
        currentLevel = int(levels[i][0])
        if currentLevel > levelMax:
            levelMax = currentLevel

    return levelMax


def reduceTree(node_id, tree):
    if tree.dict_tree[node_id].a != None:
        tree.dict_tree[node_id].x = 1
        reduce(node_id, tree)
        return tree


def reduce(node_id, tree):
    if tree.dict_tree[node_id].l.t == "Leaf" and tree.dict_tree[node_id].r.t == "Leaf":
        reduceByTree(tree.dict_tree[node_id].id, tree)

        if tree.dict_tree[node_id].x != 1:
            reduce(tree.dict_tree[node_id].p, tree)
        else:
            tree.dict_tree[node_id].x = 0
            return tree
    elif tree.dict_tree[node_id].r == "Leaf":
        reduce(tree.dict_tree[node_id].l, tree)
    else:
        reduce(tree.dict_tree[node_id].r, tree)


def reduceByTree(node_id, tree):
    tree.dict_tree[node_id].a = None

    tree.dict_tree[node_id].v = int(
        tree.dict_tree[node_id].cv[1] > tree.dict_tree[node_id].cv[0]
    )
    tree.dict_tree[node_id].t = "Leaf"
    tree.dict_tree[node_id].i = None

    tree.dict_tree[node_id].l = None
    tree.dict_tree[node_id].r = None

    del tree.dict_tree[node_id * 2]
    del tree.dict_tree[(node_id * 2) + 1]
    return tree


def countFct(data):
    x = 0
    for line in data:
        if line[2] == 0:
            x = x + 1
    y = len(data) - x
    count = [x, y]
    return count


"""
Return a list of nodes with index the level ex: nodesInLevels[0] = root, nodesInLevels[2] = nodes with level 3
"""


def listNodesinLevels(tree):
    nodesInLevels = []
    levelMax = MaxLevel(tree)
    for i in range(levelMax):
        nodesInLevels.append([])

    root = tree.getRoot()
    nodesInLevels = calculLevel(root, nodesInLevels)

    return nodesInLevels


def calculLevel(node, nodesInLevels):
    nodesInLevels[node.lv - 1].append(node)

    for i in range(len(node.c)):
        i = i + 1
        if i == 1:
            if node.l.t == "Leaf":
                nodesInLevels[node.l.lv - 1].append(node.l)
            else:
                nodesInLevels = calculLevel(node.l, nodesInLevels)

        if i == 2:
            if node.r.t == "Leaf":
                nodesInLevels[node.r.lv - 1].append(node.r)
            else:
                nodesInLevels = calculLevel(node.r, nodesInLevels)

    return nodesInLevels


"""
Function to execute
"""
Postpruning(tree, alpha, minNumP)
