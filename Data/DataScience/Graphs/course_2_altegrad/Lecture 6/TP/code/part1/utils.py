"""
Deep Learning on Graphs - ALTEGRAD - Jan 2021
"""
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def normalize_adjacency(A):
    ############## Task 1
    
    ##################
    # your code here #
    A_normalized = A + sp.identity(A.shape[0])
    D_normalized = sp.diags(1/A_normalized.dot(np.ones(A.shape[0])))
    A_normalized = D_normalized.dot(A_normalized) 
    ##################

    return A_normalized

def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    ############## Task 3
    
    ##################
    # your code here #
    idx = adj._indices()
    y_pred = []
    y = []
    pred = torch.sum(torch.mul(z[idx[0,:],:],z[idx[1,:],:]), dim=1)
    y_pred.append(pred)
    y.append(torch.ones(idx.size(1)).to(device))

    rnd_idx = torch.randint(z.size(0),idx.size())
    pred = torch.sum(torch.mul(z[rnd_idx[0,:],:],z[rnd_idx[1,:],:]), dim=1)
    y_pred.append(pred)
    y.append(torch.zeros(rnd_idx.size(1)).to(device))

    y_pred = torch.cat(y_pred, dim=0)
    y = torch.cat(y, dim=0)
    ##################
    
    loss = mse_loss(y_pred, y)
    return loss