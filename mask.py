import torch
import torch.nn as nn
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
    
from typing import Optional, Tuple

from torch import Tensor
from torch_geometric.utils import to_undirected, sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes


def mask_path(edge_index: Tensor, p: float = 0.3, walks_per_node: int = 1,
              walk_length: int = 3, num_nodes: Optional[int] = None,
              start: str = 'node',
              is_sorted: bool = False,
              training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    assert start in ['node', 'edge']
    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
    
    if not training or p == 0.0:
        return edge_index, edge_mask

    if random_walk is None:
        raise ImportError('`dropout_path` requires `torch-cluster`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    if not is_sorted:
        edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    if start == 'edge':
        sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
        start = row[sample_mask].repeat(walks_per_node)
    else:
        start = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes*p)].repeat(walks_per_node)
    
    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
    edge_mask[e_id] = False

    return edge_index[:, edge_mask], edge_index[:, ~edge_mask]


def mask_edge(edge_index: Tensor, p: float=0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')    
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]


class MaskPath(nn.Module):
    def __init__(self, p: float = 0.7, 
                 walks_per_node: int = 1,
                 walk_length: int = 3, 
                 start: str = 'node',
                 num_nodes: Optional[int]=None,
                 undirected: bool=True):
        super().__init__()
        self.p = p
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.start = start
        self.num_nodes = num_nodes
        self.undirected = undirected

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_path(edge_index, self.p,
                                                  walks_per_node=self.walks_per_node,
                                                  walk_length=self.walk_length,
                                                  start=self.start,
                                                  num_nodes=self.num_nodes)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}, walks_per_node={self.walks_per_node}, walk_length={self.walk_length}, \n"\
            f"start={self.start}, undirected={self.undirected}"


class MaskEdge(nn.Module):
    def __init__(self, p: float=0.7, undirected: bool=True):
        super().__init__()
        self.p = p
        self.undirected = undirected

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"
