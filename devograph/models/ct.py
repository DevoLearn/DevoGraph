import torch
import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity
from typing import Optional, Callable, List
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.typing import Adj
from collections.abc import Iterable
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import copy
from torch_geometric.nn.inits import glorot, zeros

class CellTrack_Model(nn.Module):
    def __init__(self,
                 hand_NodeEncoder_dic={'input_dim':13, 'fc_dims':[64, 16]},
                 learned_NodeEncoder_dic={'input_dim':13, 'fc_dims':[64, 16]},
                 intialize_EdgeEncoder_dic={'input_dim':239, 'fc_dims':[128, 64]},
                 message_passing={'in_channels': 32, 'hidden_channels':32, 'in_edge_channels':64, 
                                  'hidden_edge_channels_conv': 16, 'hidden_edge_channels_linear': [128, 64],
                                  'dropout': 0.0, 'num_layers':6, 'num_nodes_features':3},
                 edge_classifier_dic={'input_dim':64, 'fc_dims':[128, 32, 1], 'dropout_p':0.2, 'use_batchnorm':False}
                 ):
        super(CellTrack_Model, self).__init__()
        self.distance = CosineSimilarity()
        self.handcrafted_node_embedding = MLP(**hand_NodeEncoder_dic)
        self.learned_node_embedding = MLP(**learned_NodeEncoder_dic)
        self.learned_edge_embedding = MLP(**intialize_EdgeEncoder_dic)

        self.message_passing = CellTrack_GNN(**message_passing)

        self.edge_classifier = MLP(**edge_classifier_dic)

    def forward(self, x, edge_index, edge_feat):
        x1, x2 = x
        x_init = torch.cat((x1, x2), dim=-1)
        src, trg = edge_index
        similarity1 = self.distance(x_init[src], x_init[trg])
        abs_init = torch.abs(x_init[src] - x_init[trg])
        x1 = self.handcrafted_node_embedding(x1)
        x2 = self.learned_node_embedding(x2)
        x = torch.cat((x1, x2), dim=-1)
        src, trg = edge_index
        similarity2 = self.distance(x[src], x[trg])
        edge_feat_in = torch.cat((abs_init, similarity1[:, None], x[src], x[trg], torch.abs(x[src] - x[trg]), similarity2[:, None]), dim=-1)
        edge_init_features = self.learned_edge_embedding(edge_feat_in)
        edge_feat_mp = self.message_passing(x, edge_index, edge_init_features)
        pred = self.edge_classifier(edge_feat_mp).squeeze()
        return pred

class EedgePath_MPNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models dictated by the edges.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden and output sample.
        num_layers (int): Number of message passing layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 in_edge_channels: int, hidden_edge_channels: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last'):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.in_edge_channels = in_edge_channels
        self.hidden_edge_channels = hidden_edge_channels

        self.out_channels = hidden_channels
        if jk == 'cat':
            self.out_channels = num_layers * hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        from torch.nn.modules.distance import CosineSimilarity
        self.distance = CosineSimilarity()
        self.convs = ModuleList()
        self.fcs = ModuleList()

        self.jk = None
        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        self.norms = None
        if norm is not None:
            self.norms = ModuleList(
                [copy.deepcopy(norm) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if self.jk is not None:
            self.jk.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_feat: Tensor, *args, **kwargs) -> Tensor:

        src, trg = edge_index
        xs: List[Tensor] = []
        edge_features: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_feat, *args, **kwargs)

            x_src, x_trg = x[src], x[trg]
            similar = self.distance(x_src, x_trg)
            edge_feat = torch.cat((edge_feat, x_src, x_trg, torch.abs(x_src - x_trg), similar[:, None]), dim=-1)
            edge_feat = self.fcs[i](edge_feat)

            if self.norms is not None:
                x = self.norms[i](x)

            if self.act is not None:
                x = self.act(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_feat = F.dropout(edge_feat, p=self.dropout, training=self.training)

            if self.jk is not None:
                xs.append(x)

            if self.jk is not None:
                edge_features.append(edge_feat)

        return edge_feat

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')

class CellTrack_GNN(EedgePath_MPNN):
    def __init__(self,
                 in_channels: int, hidden_channels: int,
                 in_edge_channels: int, hidden_edge_channels_linear: int,
                 hidden_edge_channels_conv: int,
                 num_layers: int,
                 num_nodes_features: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels,
                         in_edge_channels, hidden_edge_channels_linear,
                         num_layers, dropout,
                         act, norm, jk)
        assert in_edge_channels == hidden_edge_channels_linear[-1]
        in_edge_dims = in_edge_channels + num_nodes_features * in_channels + 1
        self.convs.append(PDNConv(in_channels, hidden_channels, in_edge_channels,
                                  hidden_edge_channels_conv, **kwargs))
        self.fcs.append(MLP(in_edge_dims, hidden_edge_channels_linear, dropout_p=dropout))
        for _ in range(1, num_layers):
            self.convs.append(
                PDNConv(hidden_channels, hidden_channels, in_edge_channels,
                        hidden_edge_channels_conv, **kwargs))
            self.fcs.append(MLP(in_edge_dims, hidden_edge_channels_linear, dropout_p=dropout))

class PDNConv(MessagePassing):
    r"""The pathfinder discovery network convolutional operator from the
    `"Pathfinder Discovery Networks for Neural Message Passing"
    <https://arxiv.org/pdf/2010.12878.pdf>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(v) \cup
        \{i\}}f_{\Theta}(\textbf{e}_{(j,i)}) \cdot f_{\Omega}(\mathbf{x}_{j})

    where :math:`z_{i,j}` denotes the edge feature vector from source node
    :math:`j` to target node :math:`i`, and :math:`\mathbf{x}_{j}` denotes the
    node feature vector of node :math:`j`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        hidden_channels (int): Hidden edge feature dimensionality.
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 hidden_channels: int, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False)

        self.mlp = Sequential(
            Linear(edge_dim, hidden_channels),
            ReLU(inplace=True),
            Linear(hidden_channels, 1),
            Sigmoid(),
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.mlp[0].weight)
        glorot(self.mlp[2].weight)
        zeros(self.mlp[0].bias)
        zeros(self.mlp[2].bias)
        zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()

        if edge_attr is not None:
            edge_attr = self.mlp(edge_attr).squeeze(-1)

        if isinstance(edge_index, SparseTensor):
            edge_index = edge_index.set_value(edge_attr, layout='coo')

        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_attr = gcn_norm(edge_index, edge_attr,
                                                 x.size(self.node_dim), False,
                                                 self.add_self_loops)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                      False, self.add_self_loops)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_attr, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()
        if isinstance(fc_dims, Iterable):
            fc_dims = list(fc_dims)
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)

if __name__ == '__main__':
    model = CellTrack_Model()
    print(1)
