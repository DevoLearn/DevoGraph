"""Hyper-SAGNN vertex convolution layer (V2E2V message passing).

Implements the "static + dynamic embedding" attention idea from
Hyper-SAGNN, adapted to a V2E2V message-passing
layer over the sparse ``HyperedgeBatch`` incidence representation:

  1. **V2E** (vertex -> edge): for every (cell, hyperedge) incidence, build a
     *static* embedding (the cell's own representation) and a *dynamic*
     embedding (the mean of the *other* members of that hyperedge), score
     their compatibility with a small MLP, softmax-normalise per hyperedge,
     and pool the cell embeddings into one representation per hyperedge.
  2. **E2V** (edge -> vertex): every cell attends back over the
     representations of the hyperedges it belongs to (weighted by an
     edge-type embedding and the hyperedge's scalar weight), softmax-
     normalise per cell, and pools them into an update for the cell.

The result is residual-added and passed through a small feed-forward block
with LayerNorm, as in a standard Transformer block.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .incidence import HyperedgeBatch, NUM_EDGE_TYPES
from .sparse_ops import scatter_mean, scatter_sum, scatter_softmax


class HyperSAGNNLayer(nn.Module):
    """One V2E2V Hyper-SAGNN convolution layer.

    Args:
        dim: node/edge embedding dimension (input == output).
        edge_type_dim: dimension of the learned per-edge-type embedding,
            added to every edge representation before the E2V step.
        dropout: dropout applied to attention outputs and the FFN.
    """

    def __init__(self, dim: int, edge_type_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # V2E: score(static_i, dynamic_{i,e}) -> scalar
        self.v2e_score = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, 1),
        )

        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, edge_type_dim)
        self.edge_proj = nn.Linear(dim + edge_type_dim, dim)

        # E2V: score(node_i, edge_repr_e) -> scalar
        self.e2v_score = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, h: torch.Tensor, batch: HyperedgeBatch, return_attention: bool = False):
        """Args:
            h: (N, dim) node embeddings.
            batch: incidence structure of H_aug(t).
            return_attention: if True, also return a dict with the per-incidence
                ``v2e_attn`` and ``e2v_attn`` attention weights (aligned with
                ``batch.cell_index`` / ``batch.edge_index``).

        Returns:
            (N, dim) updated node embeddings, or ``(h, attn_dict)`` if
            ``return_attention`` is True.
        """
        if batch.num_edges == 0 or batch.cell_index.numel() == 0:
            if return_attention:
                return h, {"v2e_attn": torch.zeros(0), "e2v_attn": torch.zeros(0)}
            return h

        cell_index = batch.cell_index
        edge_index = batch.edge_index
        nnz = cell_index.shape[0]

        h_i = h[cell_index]  # (nnz, dim) static embedding per incidence

        # dynamic embedding: mean of *other* members of the same hyperedge
        edge_sum = scatter_sum(h_i, edge_index, batch.num_edges)  # (E, dim)
        edge_count = scatter_sum(
            torch.ones(nnz, device=h.device, dtype=h.dtype), edge_index, batch.num_edges
        )  # (E,)
        sum_per_incidence = edge_sum[edge_index]  # (nnz, dim)
        count_per_incidence = edge_count[edge_index].clamp_min(1.0)  # (nnz,)
        other_sum = sum_per_incidence - h_i
        other_count = (count_per_incidence - 1.0).clamp_min(1.0)
        dynamic = other_sum / other_count.unsqueeze(-1)
        # for singleton hyperedges (no "other" members), fall back to self
        is_singleton = (edge_count[edge_index] <= 1.0).unsqueeze(-1)
        dynamic = torch.where(is_singleton, h_i, dynamic)

        # V2E attention: pool member embeddings into one repr per hyperedge
        v2e_logits = self.v2e_score(torch.cat([h_i, dynamic], dim=-1)).squeeze(-1)  # (nnz,)
        v2e_attn = scatter_softmax(v2e_logits, edge_index, batch.num_edges)  # (nnz,)
        edge_repr = scatter_sum(h_i * v2e_attn.unsqueeze(-1), edge_index, batch.num_edges)  # (E, dim)

        # incorporate edge type + scalar weight
        type_emb = self.edge_type_emb(batch.edge_type)  # (E, edge_type_dim)
        edge_repr = self.edge_proj(torch.cat([edge_repr, type_emb], dim=-1))
        edge_repr = edge_repr * batch.edge_weight.unsqueeze(-1)

        # E2V attention: every cell pools over its incident hyperedge reprs
        edge_repr_per_incidence = edge_repr[edge_index]  # (nnz, dim)
        e2v_logits = self.e2v_score(torch.cat([h_i, edge_repr_per_incidence], dim=-1)).squeeze(-1)
        e2v_attn = scatter_softmax(e2v_logits, cell_index, batch.num_nodes)  # (nnz,)
        node_update = scatter_sum(
            edge_repr_per_incidence * e2v_attn.unsqueeze(-1), cell_index, batch.num_nodes
        )  # (N, dim)

        h_out = self.norm1(h + self.dropout(node_update))
        h_out = self.norm2(h_out + self.dropout(self.ffn(h_out)))

        if return_attention:
            return h_out, {"v2e_attn": v2e_attn.detach(), "e2v_attn": e2v_attn.detach()}
        return h_out
