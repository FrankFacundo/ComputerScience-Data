import sys
sys.path.append('..')

from collections import namedtuple

from utils.print import print2

Graph = namedtuple('Graph', ["data", "coordinates", "shape"])

row = [1, 1, 1, 2, 3, 5]
col = [4, 2, 3, 3, 4, 6]
data = [1, 1, 1, 1, 1, 1]
shape = (6, 6)

graph = Graph(data, (row, col), shape)
# print(graph.data)


def DFS(graph, start, visited=set()):
    visited.add(start)
    y_list = get_connections_from_node(graph, start)
    print2(y_list)
    for y in (y_list - visited):
        DFS(graph, y, visited=visited)
    return visited
            
def get_connections_from_node(graph, x_node):
    y_list = set()
    for x_to_y in list(zip(graph.coordinates[0], graph.coordinates[1])):
        print(x_to_y)
        if x_node in x_to_y:
            y_node = x_to_y[1 - x_to_y.index(x_node)]
            y_list.add(y_node)
    return y_list


res = DFS(graph, 1)
print2(res, id="res")

