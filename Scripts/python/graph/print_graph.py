from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_array


Graph = namedtuple('Graph', ['weight', 'coordinates', 'shape'])

row = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
col = [1, 3, 2, 4, 5, 2, 4, 5, 0, 1]
wei = [3, 1, 1, 7, 5, 4, 2, 13, 6, 8]
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


G = nx.from_scipy_sparse_array(
    graph, parallel_edges=True, create_using=nx.DiGraph)

pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, 'weight')

nx.draw_networkx(G, pos=pos, arrows=True, with_labels=True,
                 labels=labels, node_size=3000, arrowsize=30)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
plt.show()
