from torch import nn
from torch.nn import functional as F

from layers import GraphConvolution


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nhid//2)
#         self.gc3 = GraphConvolution(nhid//2, nclass)

#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc2(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc3(x, adj)

#         return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class RNN(nn.Module):
    def __init__(self, embedding_size):
        super(RNN, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)
        self.cell = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(embedding_size, 1, bias=False)

    def forward(self, x, embedding):
        mask = (x == 0)
        h = self.cell(embedding[x], src_key_padding_mask=mask.t())
        h[mask] = 0.0
        output = self.fc(h.sum(0))
        return output.squeeze()
