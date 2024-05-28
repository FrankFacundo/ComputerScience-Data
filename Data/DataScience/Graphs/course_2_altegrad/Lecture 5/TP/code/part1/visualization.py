"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import networkx as nx
import numpy as np
from deepwalk import deepwalk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Loads the web graph
G = nx.read_weighted_edgelist('./data/web_sample.edgelist', delimiter=' ', create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# a = nx.erdos_renyi_graph(50, 0.1, seed=None, directed=False)
# print(nx.is_connected(a))
# b = nx.erdos_renyi_graph(50, 0.1, seed=None, directed=False)
# print(nx.is_connected(b))
# G = nx.disjoint_union(a,b)
# print(nx.is_connected(G))
# nx.draw(G, with_labels=True, font_weight='bold')


############## Task 3
# Extracts a set of random walks from the web graph and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20

##################
model = deepwalk(G, n_walks, walk_length, n_dim)
##################


############## Task 4
# Visualizes the representations of the 100 nodes that appear most frequently in the generated walks
def visualize(model, n, dim):
    
    nodes = model.wv.index2entity[:n] # your code here
    vecs = np.empty(shape=(n, dim))
    
    ##################
    for i in range(n):
        vecs[i] = model.wv[nodes[i]]
    ##################

    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2)

    vecs_pca = my_pca.fit_transform(vecs)
    vecs_tsne = my_tsne.fit_transform(vecs_pca)

    fig, ax = plt.subplots()
    ax.scatter(vecs_tsne[:,0], vecs_tsne[:,1],s=3)
    for x, y, node in zip(vecs_tsne[:,0] , vecs_tsne[:,1], nodes):     
        ax.annotate(node, xy=(x, y), size=8)
    fig.suptitle('t-SNE visualization of node embeddings',fontsize=30)
    fig.set_size_inches(20,15)
    plt.savefig('web_embedding.pdf')
    plt.show()


visualize(model, 100, n_dim)

