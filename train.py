from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from utils import CoauthorDataset, paper_collate_fn, query_collate_fn, make_graph, make_random_adj, load_data
from models import GCN, RNN


# Training with batch adjacency matrix
data_dir = './project_data/'
paper_author = make_graph(data_dir+'paper_author.txt')
A = make_random_adj(paper_author, nodelist=np.arange(10000))
# print(A.shape)

label = torch.randn([58646])
label = label > -0.2
# print(label.float())
# print(paper_author.number_of_nodes())
# print(paper_author.number_of_edges())


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard/')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')

parser.add_argument('--epochs_pretrain', type=int, default=50, help='Number of epochs to pretrain.')
parser.add_argument('--epochs_train', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-j', '--workers', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
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

pretrain_dataset = CoauthorDataset('paper_author.txt')
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size // 2, shuffle=True, num_workers=args.workers, collate_fn=paper_collate_fn)

train_dataset = CoauthorDataset('query_public.txt')
val_dataset = CoauthorDataset('query_public.txt')
indices = list(range(len(train_dataset)))
train_sampler = SubsetRandomSampler(indices[:int(len(train_dataset) * 0.9)])
val_sampler = SubsetRandomSampler(indices[int(len(train_dataset) * 0.9):])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, collate_fn=query_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, collate_fn=query_collate_fn)

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

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(list(model.parameters()) + list(rnn.parameters()), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim.Adam(rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

if args.cuda:
    model.cuda()
    rnn.cuda()
    features = features.cuda()
    adj = adj.cuda()


def train(writer, epoch, epochs, is_pretrain=False):
    t = time.time()
    model.train()
    rnn.train()

    train_bar = tqdm(pretrain_loader) if is_pretrain else tqdm(train_loader)
    train_results = {'losses': 0.0, 'num_queries': 0}
    for queries, labels in train_bar:
        queries = queries.cuda()
        labels = labels.cuda()

        embedding = model(features, adj)
        embedding = F.pad(embedding, (0, 0, 1, 0), 'constant', 0)
        # embedding = F.pad(features, (0, 0, 1, 0), 'constant', 0)
        logits = rnn(queries, embedding)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_results['losses'] += loss.item() * queries.size(1)
        train_results['num_queries'] += queries.size(1)
        train_bar.set_description("[{}/{}] loss: {:.4f}".format(epoch, epochs, train_results['losses'] / train_results['num_queries']))

    writer.add_scalar('Loss/train', train_results['losses'] / train_results['num_queries'], epoch)
    scheduler.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        rnn.eval()

        with torch.no_grad():
            embedding = model(features, adj)
            embedding = F.pad(embedding, (0, 0, 1, 0), 'constant', 0)
            # embedding = F.pad(features, (0, 0, 1, 0), 'constant', 0)
            val_bar = tqdm(val_loader)
            val_results = {'losses': 0.0, 'correct': 0, 'num_queries': 0}
            for queries, labels in val_bar:
                queries = queries.cuda()
                labels = labels.cuda()
                logits = rnn(queries, embedding)
                loss = criterion(logits, labels)

                val_results['losses'] += loss.item() * queries.size(1)
                val_results['correct'] += torch.sum((logits > 0.0) == labels.bool()).item()
                val_results['num_queries'] += queries.size(1)
                val_bar.set_description("acc: {:4f}".format(val_results['correct'] / val_results['num_queries']))

            writer.add_scalar('Loss/validation', val_results['losses'] / val_results['num_queries'], epoch)
            writer.add_scalar('Accuracy', val_results['correct'] / val_results['num_queries'], epoch)

# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.tensorboard_dir, args.name)):
        os.makedirs(os.path.join(args.tensorboard_dir, args.name))
    if not os.path.exists(os.path.join(args.checkpoint_dir, args.name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.name))

    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, args.name))

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs_pretrain):
        train(writer, epoch + 1, args.epochs_pretrain, is_pretrain=True)
    for epoch in range(args.epochs_train):
        train(writer, epoch + 1, args.epochs_train)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, args.name, 'GCN_{:03d}.ckpt'.format(epoch)))
            torch.save(rnn.state_dict(), os.path.join(args.checkpoint_dir, args.name, 'RNN_{:03d}.ckpt'.format(epoch)))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
