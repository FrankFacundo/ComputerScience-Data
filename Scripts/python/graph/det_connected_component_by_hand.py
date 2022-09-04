import sys
sys.path.append('..')

from collections import namedtuple

from utils.print import print2

Graph = namedtuple('Graph', ["data", "coordinates", "shape"])

row = [0, 0, 0, 1, 2, 4]
col = [3, 1, 2, 2, 3, 5]
data = [1, 1, 1, 1, 1, 1]
shape = (6, 6)

graph = Graph(data, (row, col), shape)
# print(graph.data)


def DFS(graph, start, visited=None) -> set:

    def get_connections_from_node(graph, x_node):
        y_list = set()
        for x_to_y in list(zip(graph.coordinates[0], graph.coordinates[1])):
            # print(x_to_y)
            if x_node in x_to_y:
                y_node = x_to_y[1 - x_to_y.index(x_node)]
                y_list.add(y_node)
        return y_list
    
    if not visited:
        visited = set()
    visited.add(start)
    y_list = get_connections_from_node(graph, start)
    # print2(y_list, "y_list")
    # print2(visited, "visited")
    for y in (y_list - visited):
        DFS(graph, y, visited=visited)
    return visited
            
def connected_components(graph):
    non_visited = set(range(graph.shape[0]))
    groups = []
    while non_visited:
        print2(non_visited, id="non_visited")
        node_to_search = list(non_visited)[0]
        print2(node_to_search, id="node_to_search")
        group = DFS(graph, node_to_search)
        print2(group, id="group")
        groups.append(group)
        non_visited = non_visited - group
    return groups


groups = connected_components(graph)
print2(groups, id="groups")


""" print
non_visited :
{0, 1, 2, 3, 4, 5}
type: <class 'set'>


node_to_search :
0
type: <class 'int'>


group :
{0, 1, 2, 3}
type: <class 'set'>


non_visited :
{4, 5}
type: <class 'set'>


node_to_search :
4
type: <class 'int'>


group :
{4, 5}
type: <class 'set'>


groups :
[{0, 1, 2, 3}, {4, 5}]
type: <class 'list'>
"""