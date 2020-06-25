import networkx as nx
import torch 
from torch.utils.data import  TensorDataset, DataLoader

data_dir= './project_data/'


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

def get_graph_features(graph):
    ### Degree

    ### Get pagerank
    # page_rank = 
    pass

paper_author = make_graph(data_dir+'paper_author.txt')
# query_private = make_graph(data_dir+'query_prive.txt')
# query_public = make_graph(data_dir+'query_public.txt')
A = nx.adjacency_matrix(paper_author)
print(A.shape)
import numpy as np
print((A.todense()[3000]))

label = torch.randn([58646])
label = label>-0.2
print(label.float())
print(paper_author.number_of_nodes())
print(paper_author.number_of_edges())


import numpy as np
### get random adjency matirx
### input: Author Graph, Target_node_list, Adjacency matrix size:10000
def make_random_adj(graph,target_node_list, matirx_size=10000):
    # adj = adj.todense()
    random_node = np.arange(paper_author.number_of_nodes())
    random_node[target_node_list] = 0
    random_index = np.random.choice(random_node,matirx_size-len(target_node_list), replace=False) 
    random_index = np.insert(random_index, 0, target_node_list)
    output = nx.adjacency_matrix(graph, nodelist = random_index)
    return output
output_graph = make_random_adj(paper_author, [1,3,4,2]).todense()
print(output_graph.shape)

print(output_graph.sum())

class Random_Dataloader(object):
    """Face Landmarks dataset."""

    def __init__(self, file_path, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
            with open(class_path) as f:
        self.classes = f.read().splitlines()


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    


