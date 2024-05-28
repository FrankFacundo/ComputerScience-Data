"""
Influence Maximization - ALTEGRAD - Jan 2021
"""

import pandas as pd
import networkx as nx

def load_data():
    df = pd.read_excel('../data/Socrates_network.xlsx', sheet_name='Edges', header=1, usecols=[0,1])
    
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_edge(row['Vertex 1'], row['Vertex 2'])
        if index == 480:
            break
        
    return G