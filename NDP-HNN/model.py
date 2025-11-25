"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv

class HyperSAGEConv(nn.Module):
    """
    HyperSAGE: node -> hyperedge -> node aggregation.
    Paper: https://arxiv.org/abs/2010.04558
    """
    def __init__(self, in_dim, out_dim, aggr_node='mean', aggr_edge='mean',
                 dropout=0.0, bias=True):
        super().__init__()
        assert aggr_node in ('mean','max') and aggr_edge in ('mean','max')
        self.aggr_node, self.aggr_edge = aggr_node, aggr_edge
        self.lin_v = nn.Linear(in_dim,  out_dim, bias=False)
        self.lin_e = nn.Linear(out_dim, out_dim, bias=False)
        self.lin_update = nn.Linear(2*out_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _reduce(self, x, index, K, how):
        """
        x: (E,D), index: (E,), K segments.
        how: 'mean' or 'max'
        """
        if K == 0 or x.numel() == 0:
            return x.new_zeros((K, x.size(-1)))

        if how == 'max':
            out = x.new_full((K, x.size(-1)), float('-inf'))
            out.scatter_reduce_(0, index.view(-1, 1).expand_as(x), x, reduce='amax', include_self=True)
            out[out == float('-inf')] = 0.
            return out
        else:  
            out = x.new_zeros((K, x.size(-1)))
            out.scatter_reduce_(0, index.view(-1, 1).expand_as(x), x, reduce='sum', include_self=True)
            cnt = x.new_zeros((K, 1))
            ones = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
            cnt.scatter_reduce_(0, index.view(-1, 1), ones, reduce='sum', include_self=True)
            cnt = cnt.clamp_min(1.0)
            return out / cnt

    def forward(self, x, edge_index):
        row, col = edge_index
        row = row.to(torch.long).contiguous()
        col = col.to(torch.long).contiguous()

        x_proj = self.lin_v(x)  # (N, D)                         
        N = x_proj.size(0)                                      

        if edge_index.numel() == 0:
            n_agg = x_proj.new_zeros((N, x_proj.size(-1)))
            out = torch.cat([x_proj, n_agg], dim=-1)
            out = torch.relu(self.lin_update(out))
            return self.dropout(out)

        col_min = int(col.min().item())
        col = col - col_min
        M = int(col.max().item()) + 1

        e_feat = self._reduce(x_proj[row], col, M, self.aggr_node)  # (M, D)

        e_proj = self.lin_e(e_feat) # (M, D)                                     
        n_agg  = self._reduce(e_proj[col], row, N, self.aggr_edge)  # (N, D)      

        out = torch.cat([x_proj, n_agg], dim=-1)    # (N, 2D)                        
        out = torch.relu(self.lin_update(out))
        return self.dropout(out)

class UniSAGEConv(nn.Module):
    """
    UniGNN-SAGE: vertex→hyperedge→vertex with SAGE-style pooling (mean/max).
    edge_index: (2,E) incidence [node_id, hyperedge_id], 0-based per snapshot.
    ArXiv: https://arxiv.org/abs/2105.00956
    """
    def __init__(self, in_dim, out_dim, aggr_node='mean', aggr_edge='mean',
                 dropout=0.0, bias=True):
        super().__init__()
        assert aggr_node in ('mean','max') and aggr_edge in ('mean','max')
        self.aggr_node, self.aggr_edge = aggr_node, aggr_edge
        self.lin_v   = nn.Linear(in_dim,  out_dim, bias=False)   
        self.lin_e   = nn.Linear(out_dim, out_dim, bias=False)   
        self.lin_upd = nn.Linear(2*out_dim, out_dim, bias=bias)
        self.drop    = nn.Dropout(dropout)

    def _reduce(self, x, index, K, how):
        if K == 0 or x.numel() == 0:
            return x.new_zeros((K, x.size(-1)))
        if how == 'max':
            out = x.new_full((K, x.size(-1)), float('-inf'))
            out.scatter_reduce_(0, index.view(-1,1).expand_as(x), x, reduce='amax', include_self=True)
            out[out == float('-inf')] = 0.
            return out
        out = x.new_zeros((K, x.size(-1)))
        out.scatter_reduce_(0, index.view(-1,1).expand_as(x), x, reduce='sum', include_self=True)
        cnt = x.new_zeros((K,1))
        ones = torch.ones((x.size(0),1), device=x.device, dtype=x.dtype)
        cnt.scatter_reduce_(0, index.view(-1,1), ones, reduce='sum', include_self=True)
        cnt = cnt.clamp_min(1.0)
        return out / cnt

    def forward(self, x, edge_index):
        row, col = edge_index
        row = row.to(torch.long).contiguous()
        col = col.to(torch.long).contiguous()
        N = x.size(0)

        x_self = self.lin_v(x)  # (N, D)

        if edge_index.numel() == 0:
            n_agg = x_self.new_zeros((N, x_self.size(-1)))
            out = torch.cat([x_self, n_agg], dim=-1)
            return self.drop(torch.relu(self.lin_upd(out)))

        col = col - int(col.min().item())
        M = int(col.max().item()) + 1

        e_feat = self._reduce(x_self[row], col, M, self.aggr_node)  # (M, D)  

        e_proj = self.lin_e(e_feat) # (M, D)                                 
        n_agg  = self._reduce(e_proj[col], row, N, self.aggr_edge)  # (N, D)  

        out = torch.cat([x_self, n_agg], dim=-1)    # (N, 2D)                   
        return self.drop(torch.relu(self.lin_upd(out)))

class DynHNN(nn.Module):
    """
    Simpler dynamic HNN (your first variant).
    """
    def __init__(self, in_dim=4, hid_dim=64, out_dim=64, num_edge_types=2):
        super().__init__()
        self.hconvs = nn.ModuleList(
            [HypergraphConv(in_dim, hid_dim) for _ in range(num_edge_types)]
        )
        self.lin_mix = nn.Linear(num_edge_types * hid_dim, hid_dim)
        self.gru     = nn.GRUCell(hid_dim, hid_dim)
        self.readout = nn.Linear(hid_dim, out_dim)

    def forward(self, data, h_prev=None):
        outs = []
        for etype, conv in enumerate(self.hconvs):
            mask = (data.edge_attr == etype).nonzero(as_tuple=True)[0]
            ei   = data.edge_index[:, mask] if mask.numel() > 0 else data.edge_index[:, :0]
            outs.append(conv(data.x, ei))
        h = torch.relu(self.lin_mix(torch.cat(outs, dim=1)))
        h_next = h if h_prev is None else self.gru(h, h_prev)
        pred_xyz = self.readout(h_next)[:, :3]
        return h_next, pred_xyz

class DynGrowingHNN(nn.Module):
    """
    Fully-configurable dynamic hypergraph model.
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

        conv_kwargs = conv_kwargs or {}
        if isinstance(conv_type, str):
            ct = conv_type.lower()
            if ct == "hgcn":
                Conv = HypergraphConv
            elif ct == "hsage":
                Conv = lambda in_d, hid_d, **kw: HyperSAGEConv(in_d, hid_d, **kw)
            elif ct in ("ugnn"):  
                Conv = lambda in_d, hid_d, **kw: UniSAGEConv(in_d, hid_d, **kw)
            else:
                raise ValueError(f"unknown conv_type `{conv_type}`")
        else:
            Conv = conv_type

        self.hconvs = nn.ModuleList([
            Conv(in_dim, hid_dim, **conv_kwargs)
            for _ in range(num_edge_types)
        ])

        self.lin_mix = nn.Linear(num_edge_types * hid_dim, hid_dim)

        rnn_type = rnn_type.lower()
        rnn_kwargs = rnn_kwargs or {}
        self._cell_kind = rnn_type  # "rnn" | "gru" | "lstm"

        if rnn_type == "lstm":
            self.rnn = nn.LSTMCell(hid_dim, hid_dim, **rnn_kwargs)   # state: (h,c)
        elif rnn_type == "gru":
            self.rnn = nn.GRUCell(hid_dim, hid_dim, **rnn_kwargs)    # state: h
        elif rnn_type == "rnn":
            self.rnn = nn.RNNCell(hid_dim, hid_dim, **rnn_kwargs)    # state: h
        else:
            raise ValueError(f"unknown rnn_type `{rnn_type}`")

        self.use_transformer = bool(use_transformer)
        if self.use_transformer:
            d_model = hid_dim
            self.tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
                num_layers=1
            )

        self.readout = nn.Linear(hid_dim, readout_dim)

    def forward(self, data, state_prev=None):
        outs = []
        for etype, conv in enumerate(self.hconvs):
            mask = (data.edge_attr == etype).nonzero(as_tuple=True)[0]
            ei   = data.edge_index[:, mask] if mask.numel() > 0 else data.edge_index[:, :0]
            outs.append(conv(data.x, ei))
        h = torch.relu(self.lin_mix(torch.cat(outs, dim=1)))  # (N, H)

        if self.use_transformer:
            h_seq = h.unsqueeze(1)              # (N,1,H)
            h_seq = self.tf(h_seq)              # (N,1,H)
            h     = h_seq.squeeze(1)            # (N,H)

        if self._cell_kind == "lstm":
            if state_prev is None:
                h_next, c_next = self.rnn(h)
            else:
                h_prev, c_prev = state_prev
                h_next, c_next = self.rnn(h, (h_prev, c_prev))
            state_next = (h_next, c_next)
        else:
            h_next     = self.rnn(h, state_prev) if state_prev is not None else self.rnn(h)
            state_next = h_next

        out = self.readout(h_next)[:, :3]
        return state_next, out
