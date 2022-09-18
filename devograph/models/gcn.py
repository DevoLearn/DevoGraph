"""
Test GCN model
source codes link: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn.py
"""

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hid_dim, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=activation))
        # output layer
        self.layers.append(GraphConv(hid_dim, out_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h