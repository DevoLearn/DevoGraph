"""The full DHGNN model: input projection, a stack of Hyper-SAGNN V2E2V
layers with optional dynamic spatial re-clustering between layers, and a
single cell-fate prediction head.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from . import hyperedges as hyperedges_mod
from . import incidence
from .cell_universe import CellUniverse
from .hypersagnn import HyperSAGNNLayer
from .incidence import HyperedgeBatch, HyperedgeSet
from .labels import FATE_NAMES


class DynamicHypergraphStack(nn.Module):
    """Stack of Hyper-SAGNN layers with optional dynamic re-clustering.

    Between every pair of consecutive layers (when ``recluster=True``), the
    spatial hyperedge block of ``H_aug(t)`` is rebuilt by running DBSCAN on
    the *current* node embeddings of the cells present at time ``t``
    (:func:`dhgnn_lib.hyperedges.recluster_spatial_hyperedge_set`), and
    re-concatenated with the fixed lineage/functional blocks.
    This is the "dynamic hypergraph" part of DHGNN: spatial communities are
    discovered from the *learned representation*, not just raw coordinates.
    """

    def __init__(self, dim: int, num_layers: int, edge_type_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [HyperSAGNNLayer(dim, edge_type_dim=edge_type_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: HyperedgeBatch,
        universe: Optional[CellUniverse] = None,
        present_mask: Optional[torch.Tensor] = None,
        static_sets: Optional[Dict[str, HyperedgeSet]] = None,
        recluster: bool = False,
        recluster_eps: float = hyperedges_mod.SPATIAL_EPS,
        recluster_min_samples: int = hyperedges_mod.SPATIAL_MIN_SAMPLES,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer(h, batch)

            is_last = i == len(self.layers) - 1
            if recluster and not is_last and present_mask is not None and static_sets is not None:
                new_spatial = hyperedges_mod.recluster_spatial_hyperedge_set(
                    h, present_mask, universe, eps=recluster_eps, min_samples=recluster_min_samples
                )
                batch = incidence.assemble(
                    num_nodes=batch.num_nodes,
                    spatial=new_spatial.to(h.device) if new_spatial.num_edges else HyperedgeSet.empty(),
                    lineage=static_sets["lineage"],
                    functional=static_sets["functional"],
                ).to(h.device)

        return h


class DHGNNModel(nn.Module):
    """Full model: feature projection -> DynamicHypergraphStack -> fate head.

    Output:
      - ``embeddings``: (N, hidden_dim) learned per-cell representation.
      - ``fate_logits``: (N, len(FATE_NAMES)) founder-lineage classification.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_type_dim: int = 16,
        dropout: float = 0.1,
        num_fate_classes: int = len(FATE_NAMES),
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.stack = DynamicHypergraphStack(hidden_dim, num_layers, edge_type_dim=edge_type_dim, dropout=dropout)

        self.fate_head = nn.Linear(hidden_dim, num_fate_classes)

    def forward(
        self,
        x: torch.Tensor,
        batch: HyperedgeBatch,
        universe: Optional[CellUniverse] = None,
        present_mask: Optional[torch.Tensor] = None,
        static_sets: Optional[Dict[str, HyperedgeSet]] = None,
        recluster: bool = False,
    ) -> Dict[str, torch.Tensor]:
        h0 = self.input_proj(x)
        h = self.stack(
            h0,
            batch,
            universe=universe,
            present_mask=present_mask,
            static_sets=static_sets,
            recluster=recluster,
        )

        return {
            "embeddings": h,
            "fate_logits": self.fate_head(h),
        }
