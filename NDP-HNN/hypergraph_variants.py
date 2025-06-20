# -*- coding: utf-8 -*-
"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_incidence(n, hyperedges):
    """
    hyperedges: List[List[int]] 
    returns H of shape [n, len(hyperedges)]
    """
    H = x.new_zeros(n, len(hyperedges))
    for j, e in enumerate(hyperedges):
        H[e, j] = 1.
    return H

class HypergraphConv(nn.Module):
    """
    Plain hypergraph convolution (HyperGCN style).
    A = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
    h' = \sigma( A x \theta )
    Reference: Yadati, Naganand, et al. "Hypergcn: A new method for training graph convolutional networks on hypergraphs." Advances in neural information processing systems 32 (2019).
    """
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.theta = nn.Linear(in_ch, out_ch, bias=bias)

    def forward(self, x, H):
        # x: [N, F], H: [N, E]
        D_e = H.sum(0)                        # [E]
        D_v = H.sum(1)                        # [N]
        De_inv = torch.diag(1. / (D_e + 1e-6))
        Hv = H @ De_inv @ H.t()               # [N, N]
        Dv_inv_sqrt = torch.diag((D_v + 1e-6).pow(-0.5))
        A = Dv_inv_sqrt @ Hv @ Dv_inv_sqrt    # normalized
        out = A @ x                           # [N, F]
        return self.theta(out)                # [N, out_ch]

class PlainHNN(nn.Module):
    """
    2-layer plain Hypergraph Neural Network.
    """
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.conv1 = HypergraphConv(in_ch, hidden_ch)
        self.conv2 = HypergraphConv(hidden_ch, out_ch)

    def forward(self, x, H_lineage, H_spatial):
        # combine all hyperedges
        H = torch.cat([H_lineage, H_spatial], dim=1)  # [N, E1+E2]
        h = F.relu(self.conv1(x, H))
        h = self.conv2(h, H)
        return h


class HypergraphAttention(nn.Module):
    """
    A simple hypergraph attention layer (HyperSAGNN style).
    For each hyperedge e:
      y_e = \Sigma_{i\in e}(W x_i)
    Then for each node i:
      h_i' = \sigma( \Sigma_{e \in i} softmax_i( x_i^T y_e ) · y_e · \theta )
    Reference: Zhang, Ruochi, Yuesong Zou, and Jian Ma. "Hyper-SAGNN: a self-attention based graph neural network for hypergraphs." arXiv preprint arXiv:1911.02613 (2019).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.W = nn.Linear(in_ch, out_ch, bias=False)
        self.V = nn.Linear(out_ch, out_ch, bias=True)

    def forward(self, x, H):
        # x: [N,F], H: [N,E]
        N, _ = x.shape
        E = H.shape[1]
        z = self.W(x)                                # [N, out_ch]
        # hyperedge embeddings y_e
        # sum z over nodes in e, then / |e|
        e_sizes = H.sum(0).clamp(min=1).unsqueeze(1) # [E,1]
        y = (H.t() @ z) / e_sizes                    # [E, out_ch]
        # attention \alpha_{i,e} = softmax_e( z_i · y_e )
        # compute all compatibility: [N, E]
        compat = z @ y.t()                           # [N, E]
        alpha = torch.softmax(compat.masked_fill(H==0, -1e9), dim=1)
        # message: sum_e \alpha_{i,e} * y_e
        m = alpha @ y                                # [N, out_ch]
        return F.relu(self.V(m))


class AttentiveHNN(nn.Module):
    """
    2-layer hypergraph neural network with attention.
    """
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.attn1 = HypergraphAttention(in_ch, hidden_ch)
        self.attn2 = HypergraphAttention(hidden_ch, out_ch)

    def forward(self, x, H_lineage, H_spatial):
        H = torch.cat([H_lineage, H_spatial], dim=1)
        h = self.attn1(x, H)
        h = self.attn2(h, H)
        return h


class HeteroHNN(nn.Module):
    """
    Variant that treats lineage vs spatial hyperedges separately,
    and fuses their messages (concat).
    """
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.conv_lineage = HypergraphConv(in_ch, hidden_ch)
        self.conv_spatial = HypergraphConv(in_ch, hidden_ch)
        self.merge = nn.Linear(2 * hidden_ch, out_ch)

    def forward(self, x, H_lineage, H_spatial):
        h1 = F.relu(self.conv_lineage(x, H_lineage))
        h2 = F.relu(self.conv_spatial(x, H_spatial))
        h = torch.cat([h1, h2], dim=1)    # [N, 2*hidden_ch]
        return self.merge(h)             # [N, out_ch]
