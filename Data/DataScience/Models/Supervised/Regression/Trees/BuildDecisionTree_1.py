#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frank Facundo
"""
import math
from typing import List, Set, Dict, Tuple, Optional
from collections import namedtuple

INDEX_ATTR_A = 0
INDEX_ATTR_B = 1
INDEX_ATTR_C = 2

CLASS_0 = 0
CLASS_1 = 1


class Node:
    def __init__(self, attribute=None, value=None, typeNode=None, level=None, infoGain=None, constraint=None):
        """
        p : :Node: parent node
        l : :Node: left node
        r : :Node: right node

        a : :String: attribute (A or B) *if is a leave, attribute = None*
        v : :int: if is a leave, value takes C value, it means 0 or 1

        cv: :List[2] of int ex: [0, 2]: First value is the number of C=0 and the second is the number of C=1
        n : :int: number of records
        c : :int: constraint of the node
        d : :list with values of the attributes: data

        t : :String: type of node -> root(R), Intermediate(I), Leaf(L)
        lv: :int: level ex: level root = 1
        i : :float: Information gain
        """
        self.id: int = None
        self.p: Node = None
        ":Node: parent node"
        self.l: Node = None
        ":Node: left node"
        self.r: Node = None
        ":Node: right node"
        self.a = attribute
        ":String: attribute (A or B) *if is a leave, attribute = None*"
        self.v = value
        ":int: if is a leave, value takes C value, it means 0 or 1"
        self.c = constraint
        ":int: constraint of the node"
        self.cv: List[int] = None
        ":List[int]: Number of elements for each class"
        self.n: int = 0
        ":int: number of records"
        self.t: str = typeNode
        ":String: type of node -> root(R), Intermediate(I), Leaf(L)"
        self.lv: int = level
        ":int: level ex: level root = 1"
        self.i = infoGain
        ":float: Information gain"
        self.d: List[List[int]] = None
        ":list with values of the attributes: data"
        self.x = 0


class BuildDecisionTreeClass:
    """
    Build a decision tree

    -minNum -> minimum number of record per node
    BuildDecisionTree(D, A, minNum, alpha, d)
    - create a root node
    - T=Build(D, A, minNum, root, d)
    - T=PostPrune(T,alpha, minNum, d)
    - return T 
    """

    def __init__(self, file="file1.txt", minNum=2, data=None, default=0):
        self.Attr: List[str] = ["A", "B"]
        ":List[str]: Features, Inputs, X"
        self.A_Domain = [0, 1, 2, 3]
        self.B_Domain = [0, 1, 2]
        self.C_Domain = [0, 1]
        self.count = 0
        self.data: List[List[int]] = data if data else self.getTable(file)
        self.default = default
        self.dict_tree: Dict[int, Node] = {}
        ":Dict[int, Node]: Map the id of a Node with its Node"
        self.file = file
        self.minNum = minNum
        "minimum number of records per node"
        self.Process = ""
        self.Result = ""
        self.root = Node()

        self.buildRoot()
        self.build(self.data, self.Attr, self.root)

    """
    Print text of file input
    """

    def getText(self):
        f = open(self.file, "r", encoding="utf8")
        text = f.read()
        return text

    def getTable(self, file):
        with open(file, 'r') as f:
            self.data = [line.split() for line in f]

        for line in self.data:
            for i in range(len(line)):
                line[i] = int(line[i])

        return self.data

    def buildRoot(self):

        self.root.t = "Root"
        self.root.lv = 1
        self.root.cv = self.countFct(self.data)
        self.root.id = 1
        self.dict_tree[self.root.id] = self.root
        return self.root

    def countFct(self, data: List[List[int]], update=True):
        """
        Count the number of elements in each class {0, 1} 
        """
        x = 0
        for line in data:
            if line[INDEX_ATTR_C] == 0:
                x = x+1
        y = len(data)-x
        if update:
            self.count = [x, y]
        return [x, y]

    def are_rows_in_same_class(self, node: Node, data: List[List[int]]):
        """
        Check if all lines have the same class, if so, it creates a leaf and returns True, otherwise returns False.
        """
        i = 0
        while data[i][INDEX_ATTR_C] == data[0][INDEX_ATTR_C]:
            i = i+1
            if i == len(data):
                break

        if i == len(data):
            node.t = "Leaf"
            node.v = data[0][INDEX_ATTR_C]

            c = [0, 0]
            for line in data:
                if line[INDEX_ATTR_C] == CLASS_0:
                    c[0] = c[0]+1
                elif line[INDEX_ATTR_C] == CLASS_1:
                    c[1] = c[1]+1
            node.cv = c

            self.Process += "\nSame Class so it finish!"

            self.Result += "\n"
            self.Result += node.t
            self.Result += "\nLevel " + str(node.lv)
            self.Result += "\nid " + str(node.id)
            self.Result += "\nC = " + str(node.v) + "\n"

            return True
        else:
            return False

    def is_min_num(self, node, data):
        """
        Check if data has the minimum number of rows allowed.
        data : (List) list of record
        """
        if len(data) < self.minNum:
            node.t = "Leaf"

            self.countFct(data)

            if self.count[0] == self.count[1]:
                node.v = self.default
            elif self.count[0] > self.count[1]:
                node.v = 0
            elif self.count[1] > self.count[0]:
                node.v = 1

            c = [0, 0]
            for line in data:
                if line[INDEX_ATTR_C] == CLASS_0:
                    c[0] = c[0]+1
                elif line[INDEX_ATTR_C] == CLASS_1:
                    c[1] = c[1]+1
            node.cv = c

            self.Process += "\nRecords are less than minNum so it finish!"

            return True

        else:
            return False

    def is_attr_empty(self, node: Node, Attr, data):

        self.countFct(data)
        if (len(Attr) == 0):
            self.Process += "\nEmpty Attributes so it finish!"

            node.t = "Leaf"
            if (self.count[0] > self.count[1]):
                node.v = 0
            else:
                node.v = 1

            c = [0, 0]
            for line in data:
                if line[INDEX_ATTR_C] == 0:
                    c[0] = c[0]+1
                elif line[INDEX_ATTR_C] == 1:
                    c[1] = c[1]+1
            node.cv = c

            return True
        else:
            return False

    def split(self, listAttribute, data):

        maxGain = -1
        maxAttribute = None

        for Attr in listAttribute:
            best_split = self.splitAttr(Attr, data)
            if (best_split.gain > maxGain):
                maxGain = best_split.gain
                maxAttribute = Attr
                maxS = best_split
                C_values = best_split.nb_clases_by_leaf

        feature = [maxAttribute, maxS.best_split, maxGain, C_values]

        return feature

    def splitAttr(self, Attribute, data: List[List[int]]):
        splitA = [[[0], [1, 2, 3]], [[0, 1], [2, 3]],
                  [[0, 1, 2], [3]], [[0, 1, 2, 3], []]]
        splitB = [[[0], [1, 2]], [[0, 1], [2]], [[0, 1, 2], []]]
        gain = -1
        bestSplit = None

        # completing p -> number of values
        p = self.countFct(data)
        i1 = [0, 0]
        i2 = [0, 0]
        map_attr_split = {"A": splitA, "B": splitB}

        for i in map_attr_split[Attribute]:
            # Count number element for each class (O or 1) for left leaf
            i1 = [0, 0]
            # Count number element for each class (O or 1) for right leaf
            i2 = [0, 0]
            for line in data:
                if (line[INDEX_ATTR_A] in i[0]):
                    if line[INDEX_ATTR_C] == CLASS_0:
                        i1[0] = i1[0]+1
                    elif line[INDEX_ATTR_C] == CLASS_1:
                        i1[1] = i1[1]+1
                else:
                    if line[INDEX_ATTR_C] == CLASS_0:
                        i2[0] = i2[0]+1
                    elif line[INDEX_ATTR_C] == CLASS_1:
                        i2[1] = i2[1]+1

            currentGain = self.informationGain(p, i1, i2)
            if (currentGain > gain):
                bestSplit = i
                gain = currentGain
                c_values = [i1, i2]
        BestSplit = namedtuple(
            'BestSplit', ["best_split", "gain", "nb_clases_by_leaf"])
        best_split = BestSplit(bestSplit, gain, c_values)
        return best_split

    def entropy(self, node):
        n1 = node[0]
        n2 = node[1]
        n = n1 + n2

        prob_n1 = n1/n
        prob_n2 = n2/n
        log_prob_n1 = 0 if prob_n1 == 0 else math.log2(prob_n1)
        log_prob_n2 = 0 if prob_n2 == 0 else math.log2(prob_n2)

        entropy = -(prob_n1*log_prob_n1 + prob_n2*log_prob_n2)
        return entropy

    def informationGain(self, p, i_1, i_2):
        """
        p : Count the number of elements in each class {0, 1}
        """
        if (i_2 != [0, 0] and i_1 != [0, 0]):
            n1 = i_1[0] + i_1[1]
            n2 = i_2[0] + i_2[1]
            n = n1 + n2
            infoGain = self.entropy(
                p) - ((n1/n)*self.entropy(i_1) + (n2/n)*self.entropy(i_2))
            return infoGain
        else:
            return 0

    """
    leftRightSide : constraints of attribute for exemple for A it could be {0} {1,2,3}
    D1, D2 : split of the records maximizing information gain in D
    Feature : Attribute and costraint for left side
    """

    def NewData(self, data, leftRightSide, Attr):
        t = []

        for i in data:
            if (Attr == "A"):
                if (i[0] in leftRightSide):
                    t.append(i)
            elif (Attr == "B"):
                if (i[1] in leftRightSide):
                    t.append(i)
        return t

    """
    node_init : (node) node that will change
    node_end  : (node) node that will replace node_init
    """

    def setValue(self, node_end):
        self.dict_tree.update({node_end.id: node_end})
        return self

    def path(self, x):
        level = int(math.log2(x))
        listPath = []
        reste = x - 2**level
        for i in range(level):
            powers = 2 ** (2-i)
            print(reste - powers)
            if (reste - powers >= 0):
                index = 1
            else:
                index = 0
            listPath.append(index)
            reste = reste - index*powers
        return listPath

    def build(self, data: List[List[int]], listAttribute, node: Node):

        self.Process += "\n\n******************************One iteration******************************\n"

        node.d = data

        are_rows_in_same_class = self.are_rows_in_same_class(node, data)
        if are_rows_in_same_class:
            return

        is_min_num = self.is_min_num(node, data)
        if is_min_num:
            return

        is_attr_empty = self.is_attr_empty(node, listAttribute, data)
        if is_attr_empty:
            return

        # Split returns the attribute taken, the values of attributes for the left and right side and the information gain
        feature = self.split(listAttribute, data)
        self.Process += "\nFeature : " + ', '.join(map(str, feature))
        node.a = feature[0]
        node.c = feature[1]
        node.i = feature[2]

        D1 = self.NewData(data, node.c[0], node.a)

        self.Process += "\nD1 node attribute : " + node.a
        self.Process += "\nD1 node constraint : " + \
            ', '.join(map(str, node.c[0]))
        self.Process += "\nD1 : " + ', '.join(map(str, D1))

        D2 = self.NewData(data, node.c[1], node.a)

        self.Process += "\nD2 node attribute : " + node.a
        self.Process += "\nD2 node constraint : " + \
            ', '.join(map(str, node.c[1]))
        self.Process += "\nD2 : " + ', '.join(map(str, D2))

        # It remove the atribute already used
        listAttribute.remove(feature[0])
        self.Process += "\nlistAttr : " + ', '.join(map(str, listAttribute))

        for i in range(len(node.c)):
            i = i+1

            nodeChild = Node()
            if (i == 1):
                nodeChild.t = "Intermediate"
                node.l = nodeChild
                nodeChild.p = node
                node.l.lv = node.lv+1
                node.l.cv = feature[3][0]
                node.l.id = node.id * 2
                self.dict_tree[node.l.id] = node.l
                self.build(D1, listAttribute, node.l)

            if (i == 2):
                nodeChild.t = "Intermediate"
                node.r = nodeChild
                nodeChild.p = node
                node.r.lv = node.lv+1
                node.r.cv = feature[3][1]
                node.r.id = (node.id * 2) + 1
                self.dict_tree[node.r.id] = node.r
                self.build(D2, listAttribute, node.r)

    """
    Print a decision tree
    
    To this objective we start for root of level 1 and then we use breadth first search (BFS) 
    starting from the left child.
    """

    def Print_(self, process=0):
        if process == 1:
            print(self.Process)

        print(self.Result)

    def Print(self):
        levels = []
        self.Result2 = ""
        for value in self.dict_tree.values():
            if value.lv not in levels:
                levels.append(value.lv)

        for node in self.dict_tree.values():
            for i in levels:
                if (node.lv == i):
                    if node.t != "Leaf":
                        self.Result2 += "\n"
                        self.Result2 += node.t
                        self.Result2 += "\nLevel " + str(node.lv)
                        self.Result2 += "\nFeature " + \
                            str(node.a) + " " + \
                            ''.join(str(e)+" " for e in node.c[0])
                        self.Result2 += "\nid " + str(node.id)
                        self.Result2 += "\nInformation Gain " + \
                            str(node.i) + "\n"
                    else:
                        self.Result2 += "\n"
                        self.Result2 += node.t
                        self.Result2 += "\nLevel " + str(node.lv)
                        self.Result2 += "\nid " + str(node.id)
                        self.Result2 += "\nC = " + str(node.v) + "\n"

        return self.Result2

    def getResult(self):
        return self.Result

    def getRoot(self):
        return self.root


def BuildDecisionTree(file, minNum):
    TreeObject = BuildDecisionTreeClass(file, minNum)
    TreeObject.Print_(process=1)
    result_tree = TreeObject.Print()
    print('####################\n')
    print(result_tree)


"""
Function to execute
"""

BuildDecisionTree(file="file1.txt", minNum=2)
