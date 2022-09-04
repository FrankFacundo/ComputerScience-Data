import sys
sys.path.append('..')

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, coo_matrix, csr_array
import numpy as np

from utils.print import print2

row = np.array([1, 1, 1, 2, 3, 5]) - 1
col = np.array([4, 2, 3, 3, 4, 6]) - 1
data = np.array([1, 1, 1, 1, 1, 1])
shape = (6, 6)

graph = csr_matrix((data, (row, col)), shape=shape).toarray()
print2(graph, id='graph')

res = connected_components(graph, directed=False)
print2(res, id='res')

""" print
graph :
[[0 1 1 1 0 0]
 [0 0 1 0 0 0]
 [0 0 0 1 0 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 1]
 [0 0 0 0 0 0]]
type: <class 'numpy.ndarray'>


res :
(2, array([0, 0, 0, 0, 1, 1], dtype=int32))
type: <class 'tuple'>
> There are two groups: 0,1,2,3 nodes conform the first group and 5, 6 conform the second group.
"""