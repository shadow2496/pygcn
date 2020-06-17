import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx

################상현 추가#######################
def get_edge_pair(node_list):
    pairs=[]
    list_len=len(node_list)
    for i in range(list_len):
        if i == (list_len-1):
            break
        subset = node_list[i+1:]
        for item in subset:
            pairs.append((node_list[i], item))
    return pairs

# node_list=[]
def make_graph(file_path):
    file_name = open(file_path,'r' )
    g= nx.Graph()
    for num, line in enumerate(file_name.readlines()):
        if num ==0:
            node_num, edge_num =  map(int, line.split())
            for i in range(node_num):
                g.add_node(i+1)
        else:    
            node_list = list(map(int, line.split()))
            g.add_edges_from(get_edge_pair(node_list))
    assert g.number_of_nodes() == node_num 
    return g

def make_random_adj(graph,nodelist, matirx_size=10000):
    # adj = adj.todense()
    random_node = np.arange(graph.number_of_nodes())
    random_node[nodelist] = 0
    random_index = np.random.choice(random_node,matirx_size-len(nodelist), replace=False) 
    random_index = np.insert(random_index, 0, nodelist)
    output = nx.adjacency_matrix(graph, nodelist = random_index)
    return output


#################################################
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


# def load_data(path="../data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#     print(idx_features_labels)
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])

#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}


#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)

#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))

#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)

#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)

#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)

#     return adj, features, labels, idx_train, idx_val, idx_test

def load_data_sanghyeon(path="./project_data/", dataset="paper_author.txt"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    paper_author = make_graph(path+'paper_author.txt')
    print(paper_author.number_of_nodes())
    adj = nx.adjacency_matrix(paper_author)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.arange(paper_author.number_of_nodes(), dtype=np.int32)
   

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = idx
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    labels = torch.randn([58646])
    labels = labels>-0.2
    labels = labels.long()
    # labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = adj
    idx_train = torch.LongTensor(idx_train)


    return adj, features, labels, idx_train#, idx_val, idx_test

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
