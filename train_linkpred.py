import os.path as osp
import time
import argparse

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit

# custom modules
from utils import Logger, set_seed, tab_printer
from model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from edge_masking import MaskEdge, MaskPath


def train_linkpred(model, splits, args, device="cpu"):

    def train(data):
        model.train()
        loss = model.train_epoch(data.to(device), optimizer,
                                 alpha=args.alpha, batch_size=args.batch_size)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2**16):
        model.eval()
        train_data = splits['train'].to(device)
        z = model(train_data.x, train_data.edge_index)

        valid_auc, valid_ap = model.test(
            z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=batch_size)

        test_auc, test_ap = model.test(
            z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=batch_size)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    monitor = 'AUC'
    save_path = args.save_path
    loggers = {
        'AUC': Logger(args.runs, args),
        'AP': Logger(args.runs, args),
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
                if valid_result > best_valid:
                    best_valid = valid_result
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    cnt_wait = 0
                else:
                    cnt_wait += 1

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
                    print('#' * 10)
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        model.load_state_dict(torch.load(save_path))
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
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers of decoder. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.7, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.3, help='Dropout probability of decoder. (default: 0.3)')
parser.add_argument('--alpha', type=float, default=2e-3, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')


parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')

parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs. (default: 300)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument('--patience', type=int, default=10, help='(default: 10)')
parser.add_argument("--save_path", nargs="?", default="model_linkpred", help="save path for model. (default: model_linkpred)")
parser.add_argument('--debug', action='store_true', help='Whether to log information in each epoch. (default: False)')


try:
    args = parser.parse_args()
    print(tab_printer(args))
except:
    parser.print_help()
    exit(0)

if not args.save_path.endswith('.pth'):
    args.save_path += '.pth'
    
set_seed(args.seed)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])


root = osp.join('~/data/pygdata')

if args.dataset in {'arxiv', 'products'}:
    dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{args.dataset}')
    data = transform(dataset[0])
    split_idx = dataset.get_idx_split()
    data.train_nodes = split_idx['train']
    data.val_nodes = split_idx['valid']
    data.test_nodes = split_idx['test']

elif args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
    dataset = Planetoid(root, args.dataset)
    data = transform(dataset[0])

elif args.dataset == 'Reddit':
    dataset = Reddit(osp.join(root, args.dataset))
    data = transform(dataset[0])
elif args.dataset in {'Photo', 'Computers'}:
    dataset = Amazon(root, args.dataset)
    data = transform(dataset[0])
    data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
elif args.dataset in {'CS', 'Physics'}:
    dataset = Coauthor(root, args.dataset)
    data = transform(dataset[0])
    data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
else:
    raise ValueError(args.dataset)

train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=True)(data)

splits = dict(train=train_data, valid=val_data, test=test_data)


if args.mask == 'Path':
    mask = MaskPath(num_nodes=data.num_nodes)
elif args.mask == 'Edge':
    mask = MaskEdge()
else:
    mask = None

encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                           num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

print(model)

train_linkpred(model, splits, args, device=device)
