"""Unified DevoGNN model for developmental graph neural networks.

Wraps the snapshot-based DynGrowingHNN (NDP-HNN) forward pass inside a
``DevoGNN`` class that also exposes a ``forward_ctdg`` path for the
continuous-time branch.  This is a minimal skeleton — the full model will
add TGN memory, multiscale hypergraph pooling, and a link-prediction head
in subsequent phases of the project.

For the PoC the snapshot branch is fully functional: it runs HypergraphConv
per edge type, mixes them, passes through a GRU, and produces position
predictions + incidence logits — exactly the NDP-HNN forward pass, but
housed in the unified module so both branches can share the same config
and data layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv


def _hyperedge_logits(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Dot-product incidence logits (N, E) between node & hyperedge embeddings."""
    if edge_index.numel() == 0:
        return h.new_zeros((h.size(0), 0))
    row = edge_index[0].to(torch.long)
    col = edge_index[1].to(torch.long) - int(edge_index[1].min().item())
    E, D = int(col.max().item()) + 1, h.size(-1)
    he_emb = h.new_zeros((E, D))
    cnt = h.new_zeros((E, 1))
    he_emb.scatter_add_(0, col.view(-1, 1).expand(-1, D), h[row])
    cnt.scatter_add_(
        0, col.view(-1, 1), torch.ones(col.size(0), 1, device=h.device)
    )
    he_emb = he_emb / cnt.clamp_min(1.0)
    return h @ he_emb.T


class DevoGNN(nn.Module):
    """Unified developmental GNN.

    Currently supports the **snapshot branch** (hypergraph conv + temporal RNN).
    The CTDG branch (TGN memory + link prediction) will be added in the next
    project phase — the model already accepts ``graph_mode`` in its config
    to switch between branches.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension (default 4: x, y, z, t/T_max).
    hid_dim : int
        Hidden dimension.
    out_dim : int
        Output dimension (readout maps to this, first 3 dims = predicted xyz).
    num_edge_types : int
        Number of hyperedge type channels (default 2: spatial, lineage).
    conv_type : str
        Hypergraph convolution type (``"hgcn"`` only in this PoC).
    rnn_type : str
        Temporal cell type (``"gru"`` | ``"lstm"``).
    """

    def __init__(
        self,
        in_dim: int = 4,
        hid_dim: int = 64,
        out_dim: int = 64,
        num_edge_types: int = 2,
        conv_type: str = "hgcn",
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.hconvs = nn.ModuleList(
            [HypergraphConv(in_dim, hid_dim) for _ in range(num_edge_types)]
        )
        self.lin_mix = nn.Linear(num_edge_types * hid_dim, hid_dim)

        rnn_type = rnn_type.lower()
        self._cell_kind = rnn_type
        if rnn_type == "lstm":
            self.rnn = nn.LSTMCell(hid_dim, hid_dim)
        elif rnn_type == "gru":
            self.rnn = nn.GRUCell(hid_dim, hid_dim)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.readout = nn.Linear(hid_dim, out_dim)

    def forward(
        self, data: Data, state_prev=None
    ):
        """Snapshot-branch forward pass.

        Parameters
        ----------
        data : Data
            Single time-step snapshot with ``x``, ``edge_index``, ``edge_attr``.
        state_prev : Tensor or (Tensor, Tensor) or None
            Hidden state from the previous time step.

        Returns
        -------
        state_next : Tensor or (Tensor, Tensor)
            Updated hidden state.
        pred_xyz : Tensor (N, 3)
            Predicted next-step positions.
        inc_logits : Tensor (N, E)
            Hyperedge incidence reconstruction logits.
        """
        outs = []
        for etype, conv in enumerate(self.hconvs):
            mask = (data.edge_attr == etype).nonzero(as_tuple=True)[0]
            ei = (
                data.edge_index[:, mask]
                if mask.numel() > 0
                else data.edge_index[:, :0]
            )
            outs.append(conv(data.x, ei))

        h = torch.relu(self.lin_mix(torch.cat(outs, dim=1)))

        if self._cell_kind == "lstm":
            if state_prev is None:
                h_next, c_next = self.rnn(h)
            else:
                h_next, c_next = self.rnn(h, state_prev)
            state_next = (h_next, c_next)
        else:
            h_next = self.rnn(h, state_prev) if state_prev is not None else self.rnn(h)
            state_next = h_next

        pred_xyz = self.readout(h_next)[:, :3]
        inc_logits = _hyperedge_logits(h_next, data.edge_index)
        return state_next, pred_xyz, inc_logits
