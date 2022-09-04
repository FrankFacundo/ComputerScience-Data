from collections import defaultdict

edges = [[2, 1], [5, 3], [5, 1], [3, 4], [3, 1], [5, 4], [4, 1], [5, 2], [4, 2]]

node_edges = defaultdict(set)
for edge in edges:
    node_edges[edge[0]].add(edge[1])
    node_edges[edge[1]].add(edge[0])
print(node_edges)