from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils import CoauthorDataset, query_collate_fn, load_data, load_checkpoint
from models import GCN, RNN


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--pretrained_gcn', type=str, required=True)
parser.add_argument('--pretrained_rnn', type=str, required=True)

parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--feature_node', type=int, default=256, help='node_2_vec_feature_dim')
parser.add_argument('--hidden_gcn', type=int, default=200)
parser.add_argument('--feature_gcn', type=int, default=128)
parser.add_argument('--model', type=str, default='adj', help='Choosing between the adj and the node2vec')

# node2vec setting
parser.add_argument('--walk_length', type=int, default=80, help='node2vec walk_length')
parser.add_argument('--num_walks', type=int, default=10, help='node2vec num_walks')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features = load_data(args=args)
# bulid symmetric adj matrix

test_dataset = CoauthorDataset('query_private.txt')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=query_collate_fn)


# Model and optimizer
if args.model == 'adj':
    model = GCN(nfeat=features.size()[1],
                nhid=args.hidden_gcn,
                nclass=args.feature_gcn,
                dropout=args.dropout)
elif args.model == 'node2vec':
    model = GCN(nfeat=args.feature_node,
                nhid=args.hidden_gcn,
                nclass=args.feature_gcn,
                dropout=args.dropout)
rnn = RNN(args.feature_gcn)

if args.cuda:
    model.cuda()
    rnn.cuda()
    features = features.cuda()
    adj = adj.cuda()


def test():
    model.eval()
    rnn.eval()

    with torch.no_grad():
        embedding = model(features, adj)
        embedding = F.pad(embedding, (0, 0, 1, 0), 'constant', 0)
        with open('./project_data/answer_private.txt', 'w') as f:
            for queries, labels in tqdm(test_loader):
                queries = queries.cuda()
                logits = rnn(queries, embedding)
                if logits > 0.0:
                    f.write("True\n")
                else:
                    f.write("False\n")


if __name__ == '__main__':
    load_checkpoint(model, args.pretrained_gcn)
    load_checkpoint(rnn, args.pretrained_rnn)

    test()
