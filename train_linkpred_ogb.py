import argparse
from copy import copy
from tqdm.auto import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset

# custom modules
from maskgae.utils import set_seed, tab_printer
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder, DotEdgeDecoder
from maskgae.mask import MaskEdge, MaskPath


def train_linkpred(model, splits, args, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid = 0
    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    
    monitor = 'Hits@50'
    evaluator = Evaluator(name=args.dataset)    
    
    for epoch in tqdm(range(1, 1 + args.epochs)):

        loss = model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
        
        if epoch % args.eval_period == 0:
            valid_results = model.test_step_ogb(valid_data, evaluator,
                                                valid_data.pos_edge_label_index, 
                                                valid_data.neg_edge_label_index, 
                                                batch_size=batch_size)
            test_results = model.test_step_ogb(test_data, evaluator,
                                               valid_data.pos_edge_label_index, 
                                               valid_data.neg_edge_label_index, 
                                               batch_size=batch_size)            
            if valid_results[monitor] > best_valid:
                best_valid = valid_results[monitor]
                torch.save(model.state_dict(), args.save_path)
                
            print()
            print(f"Epoch {epoch} - Hits@20: {test_results['Hits@20']:.2%}", 
                  f"Hits@50: {test_results['Hits@50']:.2%}", 
                  f"Hits@100: {test_results['Hits@100']:.2%}")                   

    model.load_state_dict(torch.load(args.save_path))
    results = model.test_step_ogb(test_data, evaluator,
                                  valid_data.pos_edge_label_index, 
                                  valid_data.neg_edge_label_index, 
                                  batch_size=batch_size)  
    return results


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="ogbl-collab", help="Datasets. (default: ogbl-collab)")
parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')
parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="sage", help="GNN layer, (default: sage)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=256, help='Channels of GNN encoder. (default: 256)')
parser.add_argument('--hidden_channels', type=int, default=256, help='Channels of hidden representation. (default: 256)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.3, help='Dropout probability of encoder. (default: 0.3)')
parser.add_argument('--decoder_dropout', type=float, default=0.3, help='Dropout probability of decoder. (default: 0.3)')
parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training. (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay for training. (default: 0.)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')

parser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs. (default: 200)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-OGB.pt", help="save path for model. (default: MaskGAE-OGB.pt)")
parser.add_argument("--device", type=int, default=0)


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
    # T.ToUndirected(),
    T.ToDevice(device),
])


# (!IMPORTANT) Specify the path to your dataset directory ##############
# root = '~/public_data/pyg_data' # my root directory
root = 'data/'
########################################################################

print('Loading Data...')
if args.dataset in {'ogbl-collab'}:
    dataset = PygLinkPropPredDataset(name=args.dataset, root=root)
    data = transform(dataset[0])
    del data.edge_weight, data.edge_year
else:
    raise ValueError(args.dataset)
    
split_edge = dataset.get_edge_split()

args.year = 2010
if args.year > 0:
    year_mask = split_edge['train']['year'] >= args.year
    split_edge['train']['edge'] = split_edge['train']['edge'][year_mask]
    data.edge_index = to_undirected(split_edge['train']['edge'].t())
    print(f"{1 - year_mask.float().mean():.2%} of edges are dropped accordding to edge year {args.year}.")
    
train_data, val_data, test_data = copy(data), copy(data), copy(data)
            
args.val_as_input = True
if args.val_as_input:
    full_edge_index = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0).t()
    full_edge_index = to_undirected(full_edge_index)        

    train_data.edge_index = full_edge_index 
    val_data.edge_index = full_edge_index
    test_data.edge_index = full_edge_index
    train_data.pos_edge_label_index = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0).t()
else:
    train_data.pos_edge_label_index = split_edge['train']['edge'].t()
    

val_data.pos_edge_label_index = split_edge['valid']['edge'].t()
val_data.neg_edge_label_index = split_edge['valid']['edge_neg'].t()

test_data.pos_edge_label_index = split_edge['test']['edge'].t()
test_data.neg_edge_label_index = split_edge['test']['edge_neg'].t()

splits = dict(train=train_data, valid=val_data, test=test_data)

if args.mask == 'Path':
    mask = MaskPath(p=args.p, num_nodes=data.num_nodes, 
                    start=args.start,
                    walk_length=args.encoder_layers+1)
elif args.mask == 'Edge':
    mask = MaskEdge(p=args.p)
else:
    mask = None # vanilla GAE

encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation,
                     use_node_feats=False, node_emb=256, num_nodes=data.num_nodes)

edge_decoder = DotEdgeDecoder()

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask, 
                random_negative_sampling=True, loss='auc').to(device)

hit_20 = []
hit_50 = []
hit_100 = []

for run in range(1, args.runs+1):
    hit = train_linkpred(model, splits, args, device=device)
    hit_20.append(hit['Hits@20'])
    hit_50.append(hit['Hits@50'])
    hit_100.append(hit['Hits@100'])
    print(f"Runs {run} - Hits@20: {hit['Hits@20']:.2%}", 
          f"Hits@50: {hit['Hits@50']:.2%}", 
          f"Hits@100: {hit['Hits@100']:.2%}")    

print(f'Link Prediction Results ({args.runs} runs):\n'
      f'Hits@20: {np.mean(hit_20):.2%} ± {np.std(hit_20):.2%}',
      f'Hits@50: {np.mean(hit_50):.2%} ± {np.std(hit_50):.2%}',
      f'Hits@100: {np.mean(hit_100):.2%} ± {np.std(hit_100):.2%}',
     )