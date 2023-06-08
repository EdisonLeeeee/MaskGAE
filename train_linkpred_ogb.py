import os.path as osp
import time
import argparse
from copy import copy

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.utils import to_undirected
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset

# custom modules
from maskgae.utils import Logger, set_seed, tab_printer
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder, DotEdgeDecoder
from maskgae.mask import MaskEdge, MaskPath


def train_linkpred(model, splits, args, device="cpu"):

    def train(data):
        model.train()
        loss = model.train_epoch(data.to(device), optimizer,
                                 alpha=args.alpha, batch_size=args.batch_size)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2**16):
        model.eval()
        test_data = splits['test'].to(device)
        z = model(test_data.x, test_data.edge_index)
        results = model.test_ogb(z, splits, evaluator, batch_size=batch_size)
        return results

    evaluator = Evaluator(name=args.dataset)    
    monitor = 'Hits@50'
    save_path = args.save_path
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
    print('Start Training...')
    for run in range(args.runs):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):

            t1 = time.time()
            loss = train(splits['train'])
            t2 = time.time()

            if epoch % args.eval_period == 0:
                results = test(splits)

                valid_result = results[monitor][0]
                # if valid_result > best_valid:
                #     best_valid = valid_result
                #     best_epoch = epoch
                #     torch.save(model.state_dict(), save_path)
                #     cnt_wait = 0
                # else:
                #     cnt_wait += 1
                if args.debug:
                    for key, result in results.items():
                        valid_result, test_result = result
                        print(key)
                        print(f'Run: {run + 1:02d} / {args.runs:02d}, '
                              f'Epoch: {epoch:02d} / {args.epochs:02d}, '
                              f'Best_epoch: {best_epoch:02d}, '
                              f'Best_valid: {best_valid:.2%}%, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {valid_result:.2%}, '
                              f'Test: {test_result:.2%}',
                              f'Training Time/epoch: {t2-t1:.3f}')
                    print('#' * round(140*epoch/(args.epochs+1)))
                # if cnt_wait == args.patience:
                #     print('Early stopping!')
                #     break
        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        # model.load_state_dict(torch.load(save_path))
        results = test(splits, model)

        for key, result in results.items():
            valid_result, test_result = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Best Epoch: {best_epoch:02d}, '
                  f'Valid: {valid_result:.2%}, '
                  f'Test: {test_result:.2%}')

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print('##### Final Testing result')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


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

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 300)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument('--patience', type=int, default=30, help='(default: 30)')
parser.add_argument("--save_path", nargs="?", default="model_linkpred", help="save path for model. (default: model_linkpred)")
parser.add_argument('--debug', action='store_true', help='Whether to log information in each epoch. (default: False)')
parser.add_argument("--device", type=int, default=0)


try:
    args = parser.parse_args()
    print(tab_printer(args))
except:
    parser.print_help()
    exit(0)

if not args.save_path.endswith('.pth'):
    args.save_path += '.pth'
    
set_seed(args.seed)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    # T.ToUndirected(),
    T.ToDevice(device),
])


# root = '~/public_data/pyg_data' # my root directory
root = 'data/'

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


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask, random_negative_sampling=True, loss='auc').to(device)

print(model)

train_linkpred(model, splits, args, device=device)
