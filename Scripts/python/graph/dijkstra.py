from collections import namedtuple
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra


Graph = namedtuple('Graph', ['weight', 'coordinates', 'shape'])

row = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
col = [1, 3, 2, 4, 5, 2, 4, 5, 0, 1]
wei = [3, 1, 1, 7, 1, 4, 2, 13, 6, 8]
len = 6
labels = {
    0: 's',
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e'
}

graph = csr_array((wei, (row, col)), shape=(len, len))

dist_matrix = dijkstra(csgraph=graph, directed=True, indices=0)
print(dist_matrix)