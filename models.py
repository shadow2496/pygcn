import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

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
        return F.log_softmax(x, dim=1)


class RNN(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(RNN, self).__init__()

        self.cell = nn.LSTM(embedding_size, hidden_size, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1, bias=False)
        )

    def forward(self, x, embedding):
        input_lengths = [torch.max(x[:, i].data.nonzero()) + 1 for i in range(x.size(1))]
        sequence = pack_padded_sequence(embedding[x], lengths=input_lengths)
        _, (h_n, _) = self.cell(sequence)
        output = self.fc(h_n[-1])

        return output.squeeze()
