from distutils.command.build import build
from turtle import forward
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.nn.glob import global_add_pool, global_max_pool, global_mean_pool
from torch import Tensor
from typing import Final, Optional


def buildmlp(dim1: int,
             dim2: int,
             dim3: int,
             layer: int,
             dropout: float,
             ln: bool = True,
             tailact: bool = True):
    ret = nn.Sequential()
    for i in range(layer - 1):
        ret.append(nn.Linear(dim1, dim2))
        if ln:
            ret.append(nn.LayerNorm(dim2))
        if dropout > 0:
            ret.append(nn.Dropout(dropout, inplace=True))
        ret.append(nn.ReLU(inplace=True))
        dim1 = dim2
    ret.append(nn.Linear(dim1, dim3))
    if tailact:
        if ln:
            ret.append(nn.LayerNorm(dim3))
        if dropout > 0:
            ret.append(nn.Dropout(dropout, inplace=True))
        ret.append(nn.ReLU(inplace=True))
    return ret


class EConv(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        return scatter_add(x[edge_index[1]] * edge_attr, edge_index[0], dim=0)


from torch_geometric.utils import softmax as pygsoftmax


class AttenPool(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.k_nn = nn.Linear(hid_dim, hid_dim)
        self.qv_nn = nn.Linear(hid_dim, hid_dim)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None):
        atten = self.k_nn(x)
        atten = pygsoftmax(atten, batch, dim=-2)
        x = self.qv_nn(x)
        x = x * atten
        return x


class OutMod(nn.Module):
    def __init__(self, hid_dim: int, out_dim: int, pool: str="sum") -> None:
        super().__init__()
        if pool is None:
            self.pool = lambda x, batch: x
        elif pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            raise NotImplementedError
        self.mlp = nn.Linear(hid_dim, out_dim)

    def forward(self, x: Tensor, batch: Tensor):
        x = self.pool(x, batch)
        return self.mlp(x)


class GNN(nn.Module):
    layer: Final[int]

    def __init__(self,
                 x_dim: int,
                 ea_dim: int,
                 hid_dim: int,
                 layer: int,
                 dropout: float,
                 eamlp: Optional[nn.Module] = None,
                 **kwargs) -> None:
        super().__init__()
        if eamlp is None:
            eamlp = buildmlp(ea_dim, hid_dim, hid_dim, 2, dropout, True, True)
        self.eamlp = eamlp
        self.econv = EConv()
        self.layer = layer
        self.mlps = nn.ModuleList(
            [buildmlp(x_dim, hid_dim, hid_dim, 1, dropout, True, True)] +
            [buildmlp(hid_dim, hid_dim, hid_dim, 1, dropout, True, True) for _ in range(layer-1)])
        self.out = OutMod(hid_dim, **kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch=None):
        edge_attr = self.eamlp(edge_attr)
        for i in range(self.layer):
            x = self.mlps[i](x)
            x = x + self.econv(x, edge_index, edge_attr)
        x = self.out(x, batch)
        return x
