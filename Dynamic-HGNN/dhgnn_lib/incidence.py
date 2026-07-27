"""Sparse hyperedge containers and assembly of the identity / incidence
matrix ``H_aug(t) = [ H_spatial(t) | H_lineage | H_functional ]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

EDGE_TYPE_SPATIAL = 0
EDGE_TYPE_LINEAGE = 1
EDGE_TYPE_FUNCTIONAL = 2
NUM_EDGE_TYPES = 3
EDGE_TYPE_NAMES = ["spatial", "lineage", "functional"]


@dataclass
class HyperedgeSet:
    """A set of hyperedges over the fixed cell universe.

    Attributes:
        cell_index: LongTensor (nnz,) - node index for each (cell, edge) incidence.
        edge_index: LongTensor (nnz,) - local hyperedge index (0..num_edges-1)
            for each incidence, aligned with ``cell_index``.
        edge_weight: FloatTensor (num_edges,) - per-hyperedge scalar weight.
        num_edges: total number of hyperedges.
    """

    cell_index: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    num_edges: int

    @staticmethod
    def empty() -> "HyperedgeSet":
        return HyperedgeSet(
            cell_index=torch.zeros(0, dtype=torch.long),
            edge_index=torch.zeros(0, dtype=torch.long),
            edge_weight=torch.zeros(0, dtype=torch.float32),
            num_edges=0,
        )

    def to(self, device: torch.device) -> "HyperedgeSet":
        return HyperedgeSet(
            cell_index=self.cell_index.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            num_edges=self.num_edges,
        )


@dataclass
class HyperedgeBatch:
    """Sparse incidence representation of H_aug(t) for one timepoint.

    Attributes:
        cell_index: LongTensor (nnz,) - node (cell) index for each incidence.
        edge_index: LongTensor (nnz,) - global hyperedge index, in
            ``[0, num_edges)``, aligned with ``cell_index``.
        edge_type: LongTensor (num_edges,) - one of EDGE_TYPE_* per hyperedge.
        edge_weight: FloatTensor (num_edges,) - per-hyperedge scalar weight.
        num_nodes: total number of cells in the universe (N).
        num_edges: total number of hyperedges in H_aug(t).
    """

    cell_index: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    num_nodes: int
    num_edges: int

    def to(self, device: torch.device) -> "HyperedgeBatch":
        return HyperedgeBatch(
            cell_index=self.cell_index.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_weight=self.edge_weight.to(device),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
        )

    def edge_sizes(self) -> torch.Tensor:
        """Number of member nodes per hyperedge, shape (num_edges,)."""
        sizes = torch.zeros(self.num_edges, dtype=torch.long, device=self.cell_index.device)
        sizes.index_add_(0, self.edge_index, torch.ones_like(self.edge_index))
        return sizes

    def to_dense(self) -> torch.Tensor:
        """Dense (num_nodes, num_edges) incidence matrix (for small visualisations only)."""
        H = torch.zeros((self.num_nodes, self.num_edges), dtype=torch.float32)
        H[self.cell_index, self.edge_index] = self.edge_weight[self.edge_index]
        return H


def assemble(
    num_nodes: int,
    spatial: HyperedgeSet,
    lineage: HyperedgeSet,
    functional: HyperedgeSet,
) -> HyperedgeBatch:
    """Concatenate the three hyperedge blocks into one HyperedgeBatch."""

    blocks = [
        (spatial, EDGE_TYPE_SPATIAL),
        (lineage, EDGE_TYPE_LINEAGE),
        (functional, EDGE_TYPE_FUNCTIONAL),
    ]

    cell_index_parts: List[torch.Tensor] = []
    edge_index_parts: List[torch.Tensor] = []
    edge_type_parts: List[torch.Tensor] = []
    edge_weight_parts: List[torch.Tensor] = []

    offset = 0
    for block, etype in blocks:
        if block.num_edges > 0:
            cell_index_parts.append(block.cell_index)
            edge_index_parts.append(block.edge_index + offset)
            edge_type_parts.append(torch.full((block.num_edges,), etype, dtype=torch.long))
            edge_weight_parts.append(block.edge_weight)
        offset += block.num_edges

    cell_index = torch.cat(cell_index_parts) if cell_index_parts else torch.zeros(0, dtype=torch.long)
    edge_index = torch.cat(edge_index_parts) if edge_index_parts else torch.zeros(0, dtype=torch.long)
    edge_type = torch.cat(edge_type_parts) if edge_type_parts else torch.zeros(0, dtype=torch.long)
    edge_weight = torch.cat(edge_weight_parts) if edge_weight_parts else torch.zeros(0, dtype=torch.float32)

    return HyperedgeBatch(
        cell_index=cell_index,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        num_edges=offset,
    )
