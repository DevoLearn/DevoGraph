import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GATConv

class DynGrowingHNN(nn.Module):
    """
    a fully-configurable dynamic hypergraph (or graph) neural network.
    """
    def __init__(self,
                 in_dim=4, hid_dim=64, out_dim=64,
                 num_edge_types=2,
                 conv_type="hgcn", conv_kwargs=None,
                 rnn_type="gru",   rnn_kwargs=None,
                 use_transformer=False, transformer_kwargs=None,
                 readout_dim=None):
        super().__init__()
        readout_dim = out_dim if readout_dim is None else readout_dim

        #--- resolve conv class
        conv_kwargs = conv_kwargs or {}
        if isinstance(conv_type, str):
            ct = conv_type.lower()
            if ct == "hgcn":
                Conv = HypergraphConv
            elif ct == "gat":
                Conv = GATConv
                # default args for GAT
                default_gat = {"heads": 4, "concat": False}
                # merge user kwargs with defaults
                default_gat.update(conv_kwargs)
                conv_kwargs = default_gat
            else:
                raise ValueError(f"unknown conv_type `{conv_type}`")
        else:
            Conv = conv_type  # custom class

        self.hconvs  = nn.ModuleList([
            Conv(in_dim, hid_dim, **conv_kwargs)
            for _ in range(num_edge_types)
        ])
        self.lin_mix = nn.Linear(num_edge_types * hid_dim, hid_dim)

        #--- optional transformer
        self.use_transformer = use_transformer
        if use_transformer:
            tfkw       = transformer_kwargs or {}
            nhead      = tfkw.get("nhead", 8)
            num_layers = tfkw.get("num_layers", 1)
            layer = nn.TransformerEncoderLayer(
                d_model=hid_dim, nhead=nhead, batch_first=True,
                **{k:v for k,v in tfkw.items() if k not in {"nhead","num_layers"}}
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        #--- resolve rnn
        rnn_kwargs = rnn_kwargs or {}
        if rnn_type is None:
            self.rnn = None
        elif isinstance(rnn_type, str):
            rt = rnn_type.lower()
            if rt == "gru":
                self.rnn = nn.GRUCell(hid_dim, hid_dim, **rnn_kwargs)
            elif rt == "lstm":
                self.rnn = nn.LSTMCell(hid_dim, hid_dim, **rnn_kwargs)
            else:
                raise ValueError(f"unknown rnn_type `{rnn_type}`")
        else:
            self.rnn = rnn_type(hid_dim, hid_dim, **rnn_kwargs)

        #--- final readout
        self.readout = nn.Linear(hid_dim, readout_dim)

    def forward(self, data, state_prev=None):
        """
        forward pass
        """
        #--- conv per edge-type
        outs = []
        for etype, conv in enumerate(self.hconvs):
            mask = (data.edge_attr == etype).nonzero(as_tuple=True)[0]
            ei   = data.edge_index[:, mask]
            outs.append(conv(data.x, ei))
        h = F.relu(self.lin_mix(torch.cat(outs, dim=1)))

        #--- optional transformer mix
        if self.use_transformer:
            h = self.transformer(h.unsqueeze(0)).squeeze(0)

        #--- recurrence
        if self.rnn is None:
            h_next     = h
            state_next = None
        elif isinstance(self.rnn, nn.LSTMCell):
            if state_prev is None:
                h_next, c_next = self.rnn(h)
            else:
                h_prev, c_prev = state_prev
                h_next, c_next = self.rnn(h, (h_prev, c_prev))
            state_next = (h_next, c_next)
        else:
            h_next     = self.rnn(h, state_prev) if state_prev is not None else self.rnn(h)
            state_next = h_next

        #--- readout
        out = self.readout(h_next)[:,:3]
        return state_next, out


# #--- calling example-1: hgcn conv + gru + no transformer
# model1 = DynGrowingHNN(
#     in_dim=4, hid_dim=64, out_dim=64, num_edge_types=2,
#     conv_type="hgcn",
#     rnn_type="gru",
#     use_transformer=False
# )
# # state_next, out = model(data, state_prev=None)

# #--- calling example 2: gat conv + gru + transformer
# model2 = DynGrowingHNN(
#     in_dim=4, hid_dim=128, out_dim=64, num_edge_types=2,
#     conv_type="gat",
#     rnn_type="gru",
#     use_transformer=True, transformer_kwargs={"nhead":8, "num_layers":2}
# )
# # state_next, out = model(data, state_prev=None)
