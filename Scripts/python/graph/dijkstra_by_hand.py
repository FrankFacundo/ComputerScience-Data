from collections import namedtuple, defaultdict
import sys

import sys
sys.path.append('..')

from utils.print import print2

Graph = namedtuple('Graph',['weight', 'coordinates', 'shape'])

row = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
col = [1, 3, 2, 4, 5, 2, 4, 5, 0, 1]
wei = [3, 1, 1, 7, 1, 4, 2, 13, 6, 8]
len = 6

def get_succesors_and_edge_to_weights(graph):
    succesors = defaultdict(set)
    edge_to_weight = defaultdict(lambda: sys.maxsize)
    for edge in list(zip(graph.coordinates[0], graph.coordinates[1], graph.weight)):
        succesors[edge[0]].add(edge[1])
        edge_to_weight[(edge[0], edge[1])] = edge[2]
    return succesors, edge_to_weight

def get_new_pivot(weight_path, vertex_finished):
    # min_weight = min(weight_path.values())
    min_value = sys.maxsize
    for node, weight in weight_path.items():
        if node not in vertex_finished:
            if weight < min_value:
                min_value = weight
                pivot = node
    return pivot

def Dijkstra(graph, indice):
    vertex_finished = set([indice])
    weight_path = {}
    parent_node = defaultdict(None)

    succesors, edge_to_weight = get_succesors_and_edge_to_weights(graph)

    for index in range(graph.shape[0]):
        weight_path[index] = sys.maxsize
    weight_path[indice] = 0
    pivot = indice

    for i in range(graph.shape[0]-1):
        print2(i, id="iteration :"+str(i))
        print2(succesors[pivot], id='succesors['+str(pivot))
        print2(vertex_finished, id='vertex_finished')
        for node in (succesors[pivot] - vertex_finished):
            print2(node, id='node')
            if weight_path[node] > weight_path[pivot] + edge_to_weight[(pivot, node)]:
                weight_path[node] = weight_path[pivot] + edge_to_weight[(pivot, node)]
                parent_node[node] = pivot
            
            print2(weight_path, id="weight_path")
        pivot = get_new_pivot(weight_path, vertex_finished)
        print2(pivot, id="pivot")
        vertex_finished.add(pivot)

    return weight_path, parent_node

graph = Graph(wei, (row, col), (len,len))
weight_path, parent_node = Dijkstra(graph, 0)
print2(weight_path, id="weight_path")
print2(parent_node, id="parent_node")

""" print
iteration :0 :
0
type: <class 'int'>

succesors[0 :
{1, 3}
type: <class 'set'>

vertex_finished :
{0}
type: <class 'set'>

node :
1
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 9223372036854775807, 3: 9223372036854775807, 4: 9223372036854775807, 5: 9223372036854775807}
type: <class 'dict'>

node :
3
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 9223372036854775807, 3: 1, 4: 9223372036854775807, 5: 9223372036854775807}
type: <class 'dict'>

pivot :
3
type: <class 'int'>

iteration :1 :
1
type: <class 'int'>

succesors[3 :
{2, 4}
type: <class 'set'>

vertex_finished :
{0, 3}
type: <class 'set'>

node :
2
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 5, 3: 1, 4: 9223372036854775807, 5: 9223372036854775807}
type: <class 'dict'>

node :
4
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 5, 3: 1, 4: 3, 5: 9223372036854775807}
type: <class 'dict'>

pivot :
1
type: <class 'int'>

iteration :2 :
2
type: <class 'int'>

succesors[1 :
{2, 4}
type: <class 'set'>

vertex_finished :
{0, 1, 3}
type: <class 'set'>

node :
2
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 4, 3: 1, 4: 3, 5: 9223372036854775807}
type: <class 'dict'>

node :
4
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 4, 3: 1, 4: 3, 5: 9223372036854775807}
type: <class 'dict'>

pivot :
4
type: <class 'int'>

iteration :3 :
3
type: <class 'int'>

succesors[4 :
{0, 5}
type: <class 'set'>

vertex_finished :
{0, 1, 3, 4}
type: <class 'set'>

node :
5
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 4, 3: 1, 4: 3, 5: 16}
type: <class 'dict'>

pivot :
2
type: <class 'int'>

iteration :4 :
4
type: <class 'int'>

succesors[2 :
{5}
type: <class 'set'>

vertex_finished :
{0, 1, 2, 3, 4}
type: <class 'set'>

node :
5
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 4, 3: 1, 4: 3, 5: 5}
type: <class 'dict'>

pivot :
5
type: <class 'int'>

weight_path :
{0: 0, 1: 3, 2: 4, 3: 1, 4: 3, 5: 5}
type: <class 'dict'>

parent_node :
defaultdict(None, {1: 0, 3: 0, 2: 1, 4: 3, 5: 2})
type: <class 'collections.defaultdict'>
"""