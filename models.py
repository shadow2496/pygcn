import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence

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
    # def __init__(self, embedding_size, hidden_size):
    def __init__(self, embedding_size):

        super(RNN, self).__init__()

        # self.cell = nn.LSTM(embedding_size, hidden_size, num_layers=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)

        self.cell = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size // 4, bias=False),
            # nn.ReLU(),
            # nn.Linear(hidden_size, 1, bias=False)
            nn.Linear(embedding_size, 1, bias=True)

        )

    def forward(self, x, embedding):
        # input_lengths = [torch.max(x[:, i].data.nonzero()) + 1 for i in range(x.size(1))]
        input_lengths = [torch.tensor(25) + 1 for i in range(x.size(1))]
        # print(input_lengths)
        sequence = pack_padded_sequence(embedding[x], lengths=input_lengths)
        print(sequence[0].size)
        # print(sequence[0].shape,sequence[1] )
        # sequence = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False)
        # print(sequence.shape,'afsddafsfasd')
        # print(sequence)
        # _, (h_n, _) = self.cell(sequence)
        # output = self.fc(h_n[-1])
        output = self.cell(x)
        output = self.fc(output)
        return output.squeeze()
