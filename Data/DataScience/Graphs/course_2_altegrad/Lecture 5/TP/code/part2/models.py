"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        x = torch.mm(adj, self.fc1(x_in))
        z0 = self.relu(x)

        z0 = self.dropout(z0)

        x = torch.mm(adj, self.fc2(z0))
        z1 = self.relu(x)

        x = self.fc3(z1)

        ##################

        return F.log_softmax(x, dim=1), z1

