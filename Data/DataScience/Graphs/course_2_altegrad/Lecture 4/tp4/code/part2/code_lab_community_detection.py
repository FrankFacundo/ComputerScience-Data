"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    A = nx.adjacency_matrix(G)
    D = np.diag([G.degree[n] for n in G.nodes()])
    L_rw = np.eye(len(D)) - np.linalg.inv(D)@A
    values, U = eigs(L_rw, k=k, which="SR")
    U = U.real
    kmeans = KMeans(k, init="k-means++").fit(U)
    clustering = {}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
    ##################
    
    return clustering



############## Task 7

##################
path = "./datasets/"
graph_file = path + "CA-HepTh.txt"
G = nx.read_edgelist(graph_file, comments="#", delimiter="\t")
big_component_nodes = max(nx.connected_components(G))
sub_G = G.subgraph(big_component_nodes)
clustering = spectral_clustering(sub_G, 50)

communities = {}
for node, cluster in clustering.items():
    communities.setdefault(cluster, set()).add(int(node))
for cluster, nodes in communities.items():
    print("cluster ", cluster, ": ", len(nodes))
##################



############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    m = G.number_of_edges()
    communities = {}
    for node, cluster in clustering.items():
        communities.setdefault(cluster, set()).add(node)
    
    modularity = 0
    for cluster, nodes in communities.items():
        sub_graph = G.subgraph(list(nodes))
        lc = sub_graph.number_of_edges()
        dc = 0
        for n in nodes:
            dc += G.degree(n)
        modularity += lc/m - (dc/(2*m))**2
    ##################
    
    return modularity



############## Task 9

##################
random_clustering = {k: randint(0,49) for k, v in clustering.items()}
print("Modularity of the spectral clustering: ", modularity(sub_G, clustering))
print("Modularity of the random clustering: ", modularity(sub_G, random_clustering))
##################