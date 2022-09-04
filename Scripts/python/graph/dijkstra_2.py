import sys
sys.path.append('..')

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np

from utils.print import print2

VERBOSE = True

def findBestCity(distanceThreshold, city_nodes, city_from, city_to, city_weight):
    number_nearest_list = []
    n = city_nodes
    # minus 1 because array start at index 0 
    row = np.array(city_from) -1
    col = np.array(city_to) -1
    data = np.array(city_weight)
    graph = coo_matrix((data,(row,col)),(n,n))
    print2(graph.todense(), id="graph.todense()", activate=VERBOSE)
    print2(graph, id="graph", activate=VERBOSE)
    for i in range(n):
        dist_matrix = dijkstra(csgraph=graph, directed=False, indices=i)
        print2(dist_matrix, id="dist_matrix", activate=VERBOSE)
        # minus 1 because we have to discount the first node
        number_nearest = np.count_nonzero(dist_matrix <= distanceThreshold) - 1
        number_nearest_list.append(number_nearest)
        print2(number_nearest, id="number_nearest", activate=VERBOSE)
        print2(number_nearest_list, id="number_nearest_list", activate=VERBOSE)
    number_nearest_list = np.array(number_nearest_list)
    possible_cities = (number_nearest_list == number_nearest_list.min()).nonzero()
    print2(possible_cities, id="possible_cities", activate=VERBOSE)
    # plus 1 because we decrease the value of 1 for indexes (see line 4)
    city = possible_cities[0].max() + 1
    return city

threshold = 3
n = 3
a = [1,2]
b = [2,3]
c = [3,1]

res = findBestCity(threshold, n, a, b, c)
print(res)

""" print result
graph.todense() :
[[0 3 0]
 [0 0 1]
 [0 0 0]]
type: <class 'numpy.matrix'>

graph :
  (0, 1)	3
  (1, 2)	1
type: <class 'scipy.sparse._coo.coo_matrix'>

dist_matrix :
[0. 3. 4.]
type: <class 'numpy.ndarray'>

number_nearest :
1
type: <class 'int'>

number_nearest_list :
[1]
type: <class 'list'>

dist_matrix :
[3. 0. 1.]
type: <class 'numpy.ndarray'>

number_nearest :
2
type: <class 'int'>

number_nearest_list :
[1, 2]
type: <class 'list'>

dist_matrix :
[4. 1. 0.]
type: <class 'numpy.ndarray'>

number_nearest :
1
type: <class 'int'>

number_nearest_list :
[1, 2, 1]
type: <class 'list'>

possible_cities :
(array([0, 2]),)
type: <class 'tuple'>

3
"""