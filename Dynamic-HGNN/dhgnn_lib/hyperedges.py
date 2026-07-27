"""Builders for the three hyperedge types that make up ``H_aug(t)``.

This module is also responsible for *writing* the two new lookup tables
requested by the project proposal:

  - ``output_tables/lineage_lookup_table.csv``  (static parent/daughter triplets)
  - ``output_tables/spatial_lookup_table.csv``  (per-timepoint DBSCAN communities)

so that, alongside the pre-existing ``functional_lookup_table.csv``,
``output_tables/`` contains all three hyperedge sources needed to assemble
``H_aug(t)``.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from .cell_universe import CellUniverse, _split_members
from .incidence import HyperedgeSet

SPATIAL_EPS = 15.0
SPATIAL_MIN_SAMPLES = 3

# Dynamic re-clustering (DHGNN, Phase 4): k-means global groups + k-NN local
# neighbourhoods on the learned embeddings. Target ~15 cells per global cluster.
RECLUSTER_CLUSTER_SIZE = 15
RECLUSTER_KNN = 6


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _edges_from_member_lists(
    member_lists: Sequence[Sequence[str]],
    weights: Sequence[float],
    universe: CellUniverse,
    min_size: int = 2,
) -> HyperedgeSet:
    """Build a HyperedgeSet from a list of member-name lists.

    Hyperedges with fewer than ``min_size`` members resolvable in the cell
    universe are dropped.
    """
    cell_idx: List[int] = []
    edge_idx: List[int] = []
    edge_w: List[float] = []

    e = 0
    for members, w in zip(member_lists, weights):
        idxs = [universe.get(m) for m in members if m in universe]
        if len(idxs) < min_size:
            continue
        for i in idxs:
            cell_idx.append(i)
            edge_idx.append(e)
        edge_w.append(float(w))
        e += 1

    if e == 0:
        return HyperedgeSet.empty()

    return HyperedgeSet(
        cell_index=torch.tensor(cell_idx, dtype=torch.long),
        edge_index=torch.tensor(edge_idx, dtype=torch.long),
        edge_weight=torch.tensor(edge_w, dtype=torch.float32),
        num_edges=e,
    )


# ---------------------------------------------------------------------------
# Lineage hyperedges (static, from cells_birth_and_pos.csv)
# ---------------------------------------------------------------------------

def build_lineage_lookup_table(raw_dir: str) -> pd.DataFrame:
    """Build the lineage lookup table: one row per cell-division hyperedge.

    Each division ``{Parent Cell, Daughter 1, Daughter 2}`` becomes a static
    3-node hyperedge with weight 1.0.
    """
    cbp = pd.read_csv(os.path.join(raw_dir, "cells_birth_and_pos.csv"))

    rows = []
    for i, r in cbp.iterrows():
        rows.append(
            {
                "hyperedge_id": f"LIN_{i:04d}",
                "parent_cell": r["Parent Cell"],
                "daughter_1": r["Daughter 1"],
                "daughter_2": r["Daughter 2"],
                "members": f"{r['Parent Cell']}|{r['Daughter 1']}|{r['Daughter 2']}",
                "birth_time": r["Birth Time"],
                "parent_x": r["parent_x"],
                "parent_y": r["parent_y"],
                "parent_z": r["parent_z"],
                "hyperedge_weight": 1.0,
            }
        )

    return pd.DataFrame(rows)


def save_lineage_lookup_table(raw_dir: str, output_dir: str) -> pd.DataFrame:
    df = build_lineage_lookup_table(raw_dir)
    df.to_csv(os.path.join(output_dir, "lineage_lookup_table.csv"), index=False)
    return df


def lineage_hyperedge_set(lineage_table: pd.DataFrame, universe: CellUniverse) -> HyperedgeSet:
    member_lists = [_split_members(m) for m in lineage_table["members"]]
    weights = lineage_table["hyperedge_weight"].tolist()
    return _edges_from_member_lists(member_lists, weights, universe, min_size=2)


# ---------------------------------------------------------------------------
# Functional hyperedges (static, from existing functional_lookup_table.csv)
# ---------------------------------------------------------------------------

_FUNCTIONAL_MEMBER_COLS = ["embryo_A_sulston", "embryo_B_sulston", "embryo_C_sulston"]


def functional_hyperedge_set(functional_table: pd.DataFrame, universe: CellUniverse) -> HyperedgeSet:
    member_lists = []
    weights = []
    for _, r in functional_table.iterrows():
        members = [r[c] for c in _FUNCTIONAL_MEMBER_COLS if isinstance(r[c], str) and r[c]]
        member_lists.append(members)
        weights.append(float(r["hyperedge_weight_w_e"]))
    return _edges_from_member_lists(member_lists, weights, universe, min_size=2)


# ---------------------------------------------------------------------------
# Spatial hyperedges (dynamic, per-timepoint DBSCAN over ce_temporal_data.csv)
# ---------------------------------------------------------------------------

def build_spatial_lookup_table(
    raw_dir: str,
    eps: float = SPATIAL_EPS,
    min_samples: int = SPATIAL_MIN_SAMPLES,
) -> pd.DataFrame:
    """Run DBSCAN(eps, min_samples) on cell positions at every timepoint.

    Each non-noise cluster becomes one row / hyperedge: ``(t, cluster_id,
    members, cluster_size)``. Noise points (label == -1) are dropped.
    """
    ctd = pd.read_csv(os.path.join(raw_dir, "ce_temporal_data.csv"))

    rows = []
    eid = 0
    for t, frame in ctd.groupby("time"):
        frame = frame.drop_duplicates("cell")
        coords = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if len(frame) < min_samples:
            continue

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
        cells = frame["cell"].to_numpy()

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            members = cells[labels == cluster_id]
            rows.append(
                {
                    "hyperedge_id": f"SPA_{eid:06d}",
                    "t": int(t),
                    "cluster_id": int(cluster_id),
                    "members": "|".join(members),
                    "cluster_size": int(len(members)),
                    "hyperedge_weight": 1.0,
                }
            )
            eid += 1

    return pd.DataFrame(rows)


def save_spatial_lookup_table(raw_dir: str, output_dir: str, **kwargs) -> pd.DataFrame:
    df = build_spatial_lookup_table(raw_dir, **kwargs)
    df.to_csv(os.path.join(output_dir, "spatial_lookup_table.csv"), index=False)
    return df


def spatial_hyperedge_set(spatial_table: pd.DataFrame, t: int, universe: CellUniverse) -> HyperedgeSet:
    sub = spatial_table[spatial_table["t"] == t]
    member_lists = [_split_members(m) for m in sub["members"]]
    weights = sub["hyperedge_weight"].tolist()
    return _edges_from_member_lists(member_lists, weights, universe, min_size=2)


def recluster_spatial_hyperedge_set(
    embeddings: torch.Tensor,
    present_mask: torch.Tensor,
    universe: CellUniverse,
    eps: float = SPATIAL_EPS,
    min_samples: int = SPATIAL_MIN_SAMPLES,
) -> HyperedgeSet:
    """Rebuild spatial hyperedges by clustering the learned embeddings.

    This is the DHGNN "dynamic re-clustering" step: between
    layers, the spatial block is rebuilt from the *learned* node embeddings of
    the cells present at time ``t`` using **k-means (global communities) plus
    k-NN (local neighbourhoods)** — NOT DBSCAN. DBSCAN with a fixed ``eps``
    calibrated for raw voxel coordinates collapses to a single cluster in the
    high-dimensional, arbitrarily-scaled embedding space; k-means/k-NN are
    scale-invariant and always produce meaningful groups, as specified in the
    proposal.

    Args:
        embeddings: (N, D) node embedding tensor (any device).
        present_mask: (N,) bool tensor, True for cells present at time t.
        universe: the cell universe (indices are already aligned).
        eps, min_samples: kept for signature compatibility; unused here.
    """
    emb = embeddings.detach().cpu().numpy()
    present_idx = torch.nonzero(present_mask, as_tuple=True)[0].cpu().numpy()
    n = len(present_idx)
    if n < min_samples:
        return HyperedgeSet.empty()

    coords = emb[present_idx]
    cell_idx: List[int] = []
    edge_idx: List[int] = []
    edge_w: List[float] = []
    e = 0

    # global structure: k-means clusters
    k_global = min(max(2, n // RECLUSTER_CLUSTER_SIZE), n)
    km_labels = KMeans(n_clusters=k_global, n_init=2, max_iter=50, random_state=0).fit_predict(coords)
    for cluster_id in range(k_global):
        members = present_idx[km_labels == cluster_id]
        if len(members) < 2:
            continue
        for m in members:
            cell_idx.append(int(m))
            edge_idx.append(e)
        edge_w.append(1.0)
        e += 1

    # local structure: one hyperedge per cell's k-NN neighbourhood
    k_local = min(RECLUSTER_KNN, n - 1)
    if k_local >= 1:
        nbrs = NearestNeighbors(n_neighbors=k_local + 1).fit(coords).kneighbors(
            coords, return_distance=False
        )
        for row in nbrs:
            for m in present_idx[row]:
                cell_idx.append(int(m))
                edge_idx.append(e)
            edge_w.append(1.0)
            e += 1

    if e == 0:
        return HyperedgeSet.empty()

    return HyperedgeSet(
        cell_index=torch.tensor(cell_idx, dtype=torch.long),
        edge_index=torch.tensor(edge_idx, dtype=torch.long),
        edge_weight=torch.tensor(edge_w, dtype=torch.float32),
        num_edges=e,
    )


# ---------------------------------------------------------------------------
# Top-level store: load everything once, build H_aug(t) on demand
# ---------------------------------------------------------------------------

class HyperedgeStore:
    """Loads the three hyperedge sources and assembles ``H_aug(t)`` on demand."""

    def __init__(self, raw_dir: str, output_dir: str, universe: CellUniverse):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.universe = universe

        self.lineage_table = pd.read_csv(os.path.join(output_dir, "lineage_lookup_table.csv"))
        self.functional_table = pd.read_csv(os.path.join(output_dir, "functional_lookup_table.csv"))
        self.spatial_table = pd.read_csv(os.path.join(output_dir, "spatial_lookup_table.csv"))

        self._lineage_set = lineage_hyperedge_set(self.lineage_table, universe)
        self._functional_set = functional_hyperedge_set(self.functional_table, universe)

        self.timepoints = sorted(self.spatial_table["t"].unique().tolist())

    def spatial(self, t: int) -> HyperedgeSet:
        return spatial_hyperedge_set(self.spatial_table, t, self.universe)

    def lineage(self) -> HyperedgeSet:
        return self._lineage_set

    def functional(self) -> HyperedgeSet:
        return self._functional_set
