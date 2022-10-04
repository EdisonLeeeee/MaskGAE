import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# custom modules
from loss import info_nce_loss, ce_loss, log_rank_loss, hinge_auc_loss, auc_loss


def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(first_channels, second_channels, heads=1)
    elif name == "gat2":
        layer = GATv2Conv(first_channels, second_channels, heads=1)
    else:
        raise ValueError(name)
    return layer


def create_input_layer(num_nodes, num_node_feats,
                       use_node_feats=True, node_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if node_emb:
            emb = torch.nn.Embedding(num_nodes, node_emb)
            input_dim += node_emb
    else:
        emb = torch.nn.Embedding(num_nodes, node_emb)
        input_dim = node_emb
    return input_dim, emb


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        bn=False,
        layer="gcn",
        activation="elu",
        use_node_feats=True,
        num_nodes=None,
        node_emb=None,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats
        self.node_emb = node_emb

        if node_emb is not None and num_nodes is None:
            raise RuntimeError("Please provide the argument `num_nodes`.")

        in_channels, self.emb = create_input_layer(
            num_nodes, in_channels, use_node_feats=use_node_feats, node_emb=node_emb
        )

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels))

            self.bns.append(bn(second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, x):
        if self.use_node_feats:
            input_feat = x
            if self.node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def forward(self, x, edge_index):
        x = self.create_input_feat(x)
        edge_index = edgeidx2sparse(edge_index, x.size(0))

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat", l2_normalize=False):

        self.eval()
        assert mode in {"cat", "last"}, mode

        x = self.create_input_feat(x)
        edge_index = edgeidx2sparse(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        if l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed

        return embedding

    @torch.no_grad()
    def get_graph_embedding(self, x, edge_index, batch, l2_normalize=False, pooling='sum'):
        """https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/gin.py"""

        self.eval()
        assert batch is not None

        edge_index = edgeidx2sparse(edge_index, x.shape[0])
        xs = []  # node emb, expected to be (layers_num, batch_node_num, dim)
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        xs.append(x)

        pooling = {'sum': global_add_pool, 'mean': global_mean_pool, 'max': global_max_pool}[pooling]
        # (layers_num, batch_graph_num, dim)
        x_pool = [pooling(x, batch) for x in xs]

        # (batch_graph_num, layers_num × dim)
        embedding = torch.cat(x_pool, dim=1)
        
        if l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=1)  

        return embedding  # graph embeddings for current batch


class DotEdgeDecoder(nn.Module):
    """Simple Dot Product Edge Decoder"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_parameters(self):
        return

    def forward(self, z, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x = x.sum(-1)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class DegreeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu',
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x):

        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.mlps[-1](x)
        x = self.activation(x)

        return x


def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges


class MaskGAE(nn.Module):
    def __init__(
        self,
        encoder,
        edge_decoder,
        degree_decoder=None,
        mask=None,
        random_negative_sampling=False,
        loss="ce",
    ):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.degree_decoder = degree_decoder
        self.mask = mask

        if loss == "ce":
            self.loss_fn = ce_loss
        elif loss == "auc":
            self.loss_fn = auc_loss
        elif loss == "info_nce":
            self.loss_fn = info_nce_loss
        elif loss == "log_rank":
            self.loss_fn = log_rank_loss
        elif loss == "hinge_auc":
            self.loss_fn = hinge_auc_loss
        else:
            raise ValueError(loss)

        if random_negative_sampling:
            # this will be faster than pyg negative_sampling
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()

        if self.degree_decoder is not None:
            self.degree_decoder.reset_parameters()

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def train_epoch(
        self, data, optimizer, alpha=0.002,
        batch_size=2 ** 16, grad_norm=1.0
    ):

        x, edge_index = data.x, data.edge_index

        if self.mask is not None:
            # MaskGAE
            remaining_edges, masked_edges = self.mask(edge_index)
        else:
            # Plain GAE
            remaining_edges = edge_index
            masked_edges = getattr(data, "pos_edge_label_index", edge_index)

        loss_total = 0.0
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        for perm in DataLoader(
            range(masked_edges.size(1)), batch_size=batch_size, shuffle=True
        ):

            optimizer.zero_grad()

            z = self.encoder(x, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            # ******************* loss for edge prediction *********************
            pos_out = self.edge_decoder(
                z, batch_masked_edges, sigmoid=False
            )
            neg_out = self.edge_decoder(z, batch_neg_edges, sigmoid=False)
            loss = self.loss_fn(pos_out, neg_out)

            # ******************************************************************

            if self.degree_decoder is not None and alpha:
                deg = degree(masked_edges[1].flatten(), data.num_nodes).float()
                loss += alpha * F.mse_loss(self.degree_decoder(z).squeeze(), deg)

            loss.backward()

            if grad_norm > 0:
                # gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)

            optimizer.step()

            loss_total += loss.item()

        return loss_total

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index, batch_size=2 ** 16):

        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    @torch.no_grad()
    def test_ogb(self, z_train, z_full, splits, evaluator, batch_size=2 ** 16):

        pos_valid_edge = splits["valid"].pos_edge_label_index
        neg_valid_edge = splits["valid"].neg_edge_label_index
        pos_test_edge = splits["test"].pos_edge_label_index
        neg_test_edge = splits["test"].neg_edge_label_index

        pos_valid_pred = self.batch_predict(z_train, pos_valid_edge)
        neg_valid_pred = self.batch_predict(z_train, neg_valid_edge)

        pos_test_pred = self.batch_predict(z_full, pos_test_edge)
        neg_test_pred = self.batch_predict(z_full, neg_test_edge)

        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            valid_hits = evaluator.eval(
                {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred, }
            )[f"hits@{K}"]
            test_hits = evaluator.eval(
                {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred, }
            )[f"hits@{K}"]

            results[f"Hits@{K}"] = (valid_hits, test_hits)

        return results
