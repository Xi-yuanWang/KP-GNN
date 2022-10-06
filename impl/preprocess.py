"""
utils for processing data used for training and evaluation
"""
from inspect import stack
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
import torch
from torch_sparse import SparseTensor

def multihopkernel(data: Data, K: int, to_u: bool=True, alpha=-0.5, beta=-0.5):
    """
    Args:
        data(torch_geometric.data.Data): PyG graph data instance
    """
    assert(isinstance(data, Data))
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if to_u:
        edge_index = to_undirected(edge_index)
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg[deg==0] += 1
    ea = (deg ** alpha)[edge_index[0]]*(deg ** beta)[edge_index[0]]
    adj = SparseTensor(row = edge_index[0], col=edge_index[1], value=ea, sparse_sizes=(num_nodes, num_nodes))
    adj = adj.to_dense()
    adjl = [adj]
    for i in range(K-1):
        adjl.append(adjl[-1]@adj)
    adj = torch.stack(adjl, dim=-1)
    ei1 = torch.arange(num_nodes).unsqueeze(1).expand(-1, num_nodes).flatten()
    ei2 = torch.arange(num_nodes).unsqueeze(0).expand(num_nodes, -1).flatten()
    data.edge_index = torch.stack((ei1, ei2), dim=0)
    data.edge_attr = adj.flatten(0, 1)
    return data