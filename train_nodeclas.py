import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

# custom modules
from maskgae.utils import set_seed, tab_printer, get_dataset
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from maskgae.mask import MaskEdge, MaskPath


def train_linkpred(model, splits, args, device="cpu"):
    print('Start Training (Link Prediction Pretext Training)...')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid = 0
    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    
    model.reset_parameters()
    
    for epoch in tqdm(range(1, 1 + args.epochs)):

        loss = model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
        
        if epoch % args.eval_period == 0:
            valid_auc, valid_ap = model.test_step(valid_data, 
                                                  valid_data.pos_edge_label_index, 
                                                  valid_data.neg_edge_label_index, 
                                                  batch_size=batch_size)
            if valid_auc > best_valid:
                best_valid = valid_auc
                best_epoch = epoch
                torch.save(model.state_dict(), args.save_path)

    model.load_state_dict(torch.load(args.save_path))
    test_auc, test_ap = model.test_step(test_data, 
                                        test_data.pos_edge_label_index, 
                                        test_data.neg_edge_label_index, 
                                        batch_size=batch_size)   
    
    print(f'Link Prediction Pretraining Results:\n'
          f'AUC: {test_auc:.2%}',
          f'AP: {test_ap:.2%}')
    return test_auc, test_ap

def train_nodeclas(model, data, args, device='cpu'):
    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()

    if args.dataset in {'arxiv', 'products', 'mag'}:
        batch_size = 4096
    else:
        batch_size = 512
        
    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)

    data = data.to(device)
    y = data.y.squeeze()
    embedding = model.encoder.get_embedding(data.x, data.edge_index)

    if args.l2_normalize:
        embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed    

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    print('Start Training (Node Classification)...')
    results = []
    
    for run in range(1, args.runs+1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(), 
                                     lr=0.01, 
                                     weight_decay=args.nodeclas_weight_decay)

        best_val_metric = test_metric = 0
        for epoch in tqdm(range(1, 101), desc=f'Training on runs {run}...'):
            clf.train()
            for nodes in train_loader:
                optimizer.zero_grad()
                loss_fn(clf(embedding[nodes]), y[nodes]).backward()
                optimizer.step()
                
            val_metric, test_metric = test(val_loader), test(test_loader)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
        results.append(best_test_metric)
        print(f'Runs {run}: accuracy {best_test_metric:.2%}')
                          
    print(f'Node Classification Results ({args.runs} runs):\n'
          f'Accuracy: {np.mean(results):.2%} ± {np.std(results):.2%}')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder layers. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size for link prediction training. (default: 2**16)')

parser.add_argument("--start", nargs="?", default="node", help="Which Type to sample starting nodes for random walks, (default: node)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 500)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=30, help='(default: 30)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-NodeClas.pt", help="save path for model. (default: MaskGAE-NodeClas.pt)")
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--full_data', action='store_true', help='Whether to use full data for pretraining. (default: False)')


try:
    args = parser.parse_args()
    print(tab_printer(args))
except:
    parser.print_help()
    exit(0)

set_seed(args.seed)

if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])


# (!IMPORTANT) Specify the path to your dataset directory ##############
# root = '~/public_data/pyg_data' # my root directory
root = 'data/'
########################################################################

data = get_dataset(root, args.dataset, transform=transform)

train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=False)(data)
if args.full_data:
    # Use full graph for pretraining
    splits = dict(train=data, valid=val_data, test=test_data)
else:
    splits = dict(train=train_data, valid=val_data, test=test_data)


if args.mask == 'Path':
    mask = MaskPath(p=args.p, 
                    num_nodes=data.num_nodes, 
                    start=args.start,
                    walk_length=args.encoder_layers+1)
elif args.mask == 'Edge':
    mask = MaskEdge(p=args.p)
else:
    mask = None # vanilla GAE

encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                           num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

train_linkpred(model, splits, args, device=device)
train_nodeclas(model, data, args, device=device)
