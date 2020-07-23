import os
import pickle
import random

import networkx as nx
from node2vec import Node2Vec
import numpy as np
from scipy import sparse as sp
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CoauthorDataset(Dataset):
    def __init__(self, filename, dataset_dir='./project_data/'):
        super(CoauthorDataset, self).__init__()

        self.load_labels = False
        self.queries = []
        SOS = 58647
        EOS = 58648

        with open(os.path.join(dataset_dir, filename), 'r') as f:
            for line in f.readlines()[1:]:
                query = list(map(int, line.split()))
                # random.shuffle(query)
                # self.queries.append([SOS, *query, EOS])
                self.queries.append(query)

        if filename == 'query_public.txt':
            with open(os.path.join(dataset_dir, 'answer_public.txt'), 'r') as f:
                self.labels = [float(line.strip() == 'True') for line in f]
        else:
            self.labels = [True] * len(self.queries)

    def __getitem__(self, index):
        return self.queries[index], self.labels[index]

    def __len__(self):
        return len(self.queries)


def paper_collate_fn(batched_samples):
    weight = [91240, 31701, 8651, 2662, 1087, 588, 384, 262, 215, 184, 155, 128, 121, 80, 96, 60, 58, 60, 65, 50, 24, 24, 36, 27]
    lengths = random.choices(range(2, 26), weight, k=len(batched_samples))
    batched_samples.extend([(sorted(random.sample(range(1, 58647), k=length)), 0.0) for length in lengths])
    random.shuffle(batched_samples)

    batched_samples = sorted(batched_samples, key=lambda t: len(t[0]), reverse=True)
    queries = pad_sequence([torch.tensor(sample[0]) for sample in batched_samples])

    if batched_samples[0][1] is not None:
        labels = torch.tensor([sample[1] for sample in batched_samples])
        return queries, labels
    else:
        return queries, None


def query_collate_fn(batched_samples):
    batched_samples = sorted(batched_samples, key=lambda t: len(t[0]), reverse=True)
    queries = pad_sequence([torch.tensor(sample[0]) for sample in batched_samples])

    if batched_samples[0][1] is not None:
        labels = torch.tensor([sample[1] for sample in batched_samples])
        return queries, labels
    else:
        return queries, None


#######################################
def get_edge_pair(node_list):
    pairs = []
    list_len = len(node_list)
    for i in range(list_len):
        if i == (list_len-1):
            break
        subset = node_list[i+1:]
        for item in subset:
            pairs.append((node_list[i], item))
    return pairs


# node_list=[]
def make_graph(file_path):
    file_name = open(file_path, 'r')
    g = nx.Graph()
    lines = file_name.readlines()
    for num, line in enumerate(lines[:int((len(lines) - 1) * 0.7) + 1]):
        if num == 0:
            node_num, edge_num = map(int, line.split())
            for i in range(node_num+2):
                g.add_node(i+1)
        else:    
            node_list = list(map(int, line.split()))
            g.add_edges_from(get_edge_pair(node_list))
    assert g.number_of_nodes() == node_num+2
    return g


def make_random_adj(graph, nodelist, matirx_size=10000):
    # adj = adj.todense()
    random_node = np.arange(graph.number_of_nodes())
    random_node[nodelist] = 0
    random_index = np.random.choice(random_node, matirx_size-len(nodelist), replace=False)
    random_index = np.insert(random_index, 0, nodelist)
    output = nx.adjacency_matrix(graph, nodelist=random_index)
    return output
#################################################


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

####여기에 feautres를 더 추가해줄 여력이 있을 듯 하긴 함...
# def get_features(g):
    # Degree = g.degree()
    # page_rank = nx.pagerank(g, alpha = 0.85)
    # between_centraily = nx.edge_betweenness_centrality(g)
    # simrank = nx.simrank_similarity(g)
    # katz = nx.katz_centrality(g)
    # print(simrank)
# nx.relabel_nodes(G, { n:str(n) for n in G.nodes()})


def load_data(args, path="./project_data/", dataset="paper_author.txt"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    paper_author = make_graph(path+dataset)
    print('The number of nodes :', paper_author.number_of_nodes())
    adj = nx.adjacency_matrix(paper_author)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if args.model == 'adj':
        features = adj

    elif args.model == 'node2vec':
        print('Already exist Node2vec file')
        file_name = './Node2vec_walk_%s_num_walks_%s_truncated.pickle' % (str(args.walk_length), str(args.num_walks))
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as file:
                features = pickle.load(file)
        else:
            node2vec = Node2Vec(graph=paper_author,  # target graph
                                dimensions=int(args.feature_node),  # embedding dimension
                                walk_length=int(args.walk_length),  # number of nodes in each walks
                                p=2,  # return hyper parameter
                                q=1,  # inout parameter, q값을 작게 하면 structural equivalence를 강조하는 형태로 학습됩니다.
                                weight_key=None,  # if weight_key in attrdict
                                num_walks=int(args.num_walks), 
                                workers=4,
                                )
            features = torch.tensor(node2vec.fit(window=10, min_count=0).wv.vectors)
            with open(file_name, 'wb') as file:
                pickle.dump(features, file)
    return adj, features  # , labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == "__main__":
    adj, features, labels, idx_train = load_data_sanghyeon()
