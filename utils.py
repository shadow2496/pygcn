import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from node2vec import Node2Vec
import pickle
import os

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

####여기에 feautres를 더 추가해줄 여력이 있을 듯 하긴 함...
def get_features(g):
    # Degree = g.degree()
    # page_rank = nx.pagerank(g, alpha = 0.85)
    # between_centraily = nx.edge_betweenness_centrality(g)
    simrank = nx.simrank_similarity(g)

    # katz = nx.katz_centrality(g)
    print(simrank)

# nx.relabel_nodes(G, { n:str(n) for n in G.nodes()})



def load_data_sanghyeon(args,path="./project_data/", dataset="paper_author.txt"):
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

    if args.model =='adj':
        features = adj

    elif args.model =='node2vec':
        print('Already exist Node2vec file')
        file_name = './Node2vec_walk_%s_num_walks_%s.pickle'%(str(args.walk_length),str(args.num_walks))
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as file:
                node2vec=pickle.load(file)
        else:
            node2vec = Node2Vec(graph=paper_author, # target graph
                                dimensions=int(args.features), # embedding dimension
                                walk_length=int(args.walk_length), # number of nodes in each walks 
                                p = 1, # return hyper parameter
                                q = 0.0001, # inout parameter, q값을 작게 하면 structural equivalence를 강조하는 형태로 학습됩니다. 
                                weight_key=None, # if weight_key in attrdict 
                                num_walks=int(args.num_walks), 
                                workers=4,
                            )
            with open(file_name, 'wb') as file:
                pickle.dump(node2vec, file)
        features = torch.tensor(node2vec.fit(window=1,min_count=0).wv.vectors)


    # get_features(paper_author)
    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.arange(paper_author.number_of_nodes(), dtype=np.int32)
    idx_train = idx
    #### For test, 나중에 지울거임 (label 안 받게)
    labels = torch.randn([58646])
    labels = labels>-0.2
    labels = labels.long()
    # labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    print(features)


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

if __name__ == "__main__":
    adj, features, labels, idx_train = load_data_sanghyeon()
