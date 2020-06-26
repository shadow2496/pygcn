from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from pygcn.models import GCN

import networkx as nx
import torch 
from scipy.sparse import coo_matrix
import scipy.sparse as sp



### Training with batch adjacency matirx
data_dir= './project_data/'
paper_author = make_graph(data_dir+'paper_author.txt')
A = make_random_adj(paper_author,nodelist = np.arange(10000))
# print(A.shape)

label = torch.randn([58646])
label = label>-0.2
# print(label.float())
# print(paper_author.number_of_nodes())
# print(paper_author.number_of_edges())


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=200,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--features', type=int, default=256,
                    help='node_2_vec_feature_dim')
parser.add_argument('--model', type=str, default='adj',
                    help='Choosing between the adj and the node2vec')

parser.add_argument('--load_node2vec', type=bool, default=True,
                    help='load precalculated node2vec object')

#### node2vec setting
parser.add_argument('--walk_length', type=int, default=50,
                    help=' node2vec walk_length')
parser.add_argument('--num_walks', type=int, default=4,
                    help=' node2vec num_walks')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



# Load data
adj, features = load_data_(args = args)
# bulid symmetric adj matrix

print(adj.shape, features.shape)

# Model and optimizer
if args.model =='adj':
    model = GCN(nfeat=features.size()[1],
                nhid=args.hidden,
                nclass=2,
                dropout=args.dropout)
elif args.model =='node2vec':
    model = GCN(nfeat=args.features,
                nhid=args.hidden,
                nclass=2,
                # nclass=labels.max().item() + 1,
                dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
        #   'loss_val: {:.4f}'.format(loss_val.item()),
        #   'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
