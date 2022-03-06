"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
# from dgl.nn import GraphConv
from .graphconv_edge_weight import GraphConvEdgeWeight as GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        #! 因为这里前后有两个input_layer和output_layer, 它们也是代表着hidden layer的
        #! 所以实际的n_layers要减去它们两个
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=normalization, allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=normalization, allow_zero_in_degree=True))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, norm=normalization, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)
    #! 这边传的时候和layer的顺序不同
    def forward(self, features, g, edge_weight):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            #! 在forward 的时候，传的是图，feature, weight
            h = layer(g, h, edge_weights=edge_weight)
        return h