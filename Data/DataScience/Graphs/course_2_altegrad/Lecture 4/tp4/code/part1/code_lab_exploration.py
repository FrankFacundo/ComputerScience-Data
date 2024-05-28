"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
path = "./datasets/"
graph_file = path + "CA-HepTh.txt"
G = nx.read_edgelist(graph_file, comments="#", delimiter="\t")
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
##################



############## Task 2

##################
n_connected_comp = nx.number_connected_components(G) 
print("number of connected components: ", n_connected_comp)
connected_comp_list = nx.connected_components(G)
big_component_nodes = max(connected_comp_list)
sub_G = G.subgraph(big_component_nodes)
print("Number of nodes of big component: ", sub_G.number_of_nodes())
print("Nodes ratio: ", sub_G.number_of_nodes()/G.number_of_nodes())
print("Number of edges of big component: ", sub_G.number_of_edges())
print("Edges ratio: ", sub_G.number_of_edges()/G.number_of_edges())
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
print("Min degree: ", np.min(degree_sequence))
print("Max degree: ", np.max(degree_sequence))
print("Mean degree: ", np.mean(degree_sequence))
print("Median degree: ", np.median(degree_sequence))
##################



############## Task 4
degree_freq = nx.degree_histogram(G)
plt.figure()
plt.plot(range(len(degree_freq)), degree_freq)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Freq")

plt.figure()
plt.loglog(range(len(degree_freq)), degree_freq)
plt.title("Log-log Degree Freq")
plt.xlabel("Log-Degree")
plt.ylabel("Log-Frequency")
plt.show()
##################

##################




############## Task 5

##################
cluster_coef = nx.transitivity(G)
print("Global Clustering coef:", cluster_coef)
##################