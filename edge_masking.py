import torch
import torch.nn as nn
import torch_cluster
from torch_geometric.utils import to_undirected


def mask_path(edge_index,
              walks_per_node: int = 2,
              walk_length: int = 4, r: float = 0.5,
              p: float = 1, q: float = 1, num_nodes=None,
              by='degree',
              replacement=True):

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    assert by in {'degree', 'uniform'}

    row = edge_index[0]
    col = edge_index[1]
    deg = torch.zeros(num_nodes, device=row.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=row.device))

    if isinstance(r, float):
        assert 0 < r <= 1
        num_starts = int(r * num_nodes)
        if by == 'degree':
            prob = deg.float() / deg.sum()
            start = prob.multinomial(num_samples=num_starts, replacement=replacement)
        else:
            start = torch.randperm(num_nodes, device=edge_index.device)[:num_starts]
    elif torch.is_tensor(r):
        start = r.to(edge_index)
        n = start.size(0)
        start = start[torch.randperm(n)[:n // 3]]
    else:
        raise ValueError(r)

    if walks_per_node:
        start = start.repeat(walks_per_node)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)

    mask = row.new_ones(row.size(0), dtype=torch.bool)
    mask[e_id.view(-1)] = False
    return edge_index[:, mask], edge_index[:, e_id]


def mask_edge(edge_index, p=0.7):
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]


class MaskPath(nn.Module):
    def __init__(self, walks_per_node: int = 2,
                 walk_length: int = 4, r: float = 0.5,
                 p: float = 1, q: float = 1, num_nodes=None,
                 by='degree', undirected=True, replacement=True):
        super(MaskPath, self).__init__()
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.r = r
        self.num_nodes = num_nodes
        self.by = by
        self.undirected = undirected
        self.replacement = replacement

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_path(edge_index,
                                                  walks_per_node=self.walks_per_node,
                                                  walk_length=self.walk_length,
                                                  p=self.p, q=self.q,
                                                  r=self.r, num_nodes=self.num_nodes, by=self.by, replacement=self.replacement)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"walks_per_node={self.walks_per_node}, walk_length={self.walk_length}, \n"\
            f"r={self.r}, p={self.p}, q={self.q}, by={self.by}, undirected={self.undirected}, replacement={self.replacement}"


class MaskEdge(nn.Module):
    def __init__(self, p=0.7, undirected=True):
        super(MaskEdge, self).__init__()
        self.p = p
        self.undirected = undirected

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"
