"""
Tests for NDP-HNN snapshot construction.

Covers the fix for edge_attr/edge_index misalignment in build_snapshots():
  - edge_attr must have one element per node-hyperedge membership (per edge_index column)
  - type-based masks must collectively cover all edge_index columns
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import networkx as nx
import pytest

from snapshots import build_snapshots


def make_synthetic_dataset():
    """
    Minimal synthetic dataset with 4 cells across 2 timesteps.

      t=0: A (0,0,0), B (5,0,0)          — 2 alive nodes
      t=1: A (0,0,0), B (5,0,0),          — 4 alive nodes
           C (0.5,0.5,0), D (5.5,0.5,0)

    Lineage: A -> C, A -> D
    With k=2 and large spatial_radius every alive set forms
    multi-node spatial hyperedges, and C+D form a lineage hyperedge at t=1.
    This guarantees hyperedges with >1 node, which is required to expose
    the pre-fix shape mismatch.
    """
    cells = sorted(['A', 'B', 'C', 'D'])
    idx = {c: i for i, c in enumerate(cells)}

    birth_feat = np.array([
        [0.0, 0.0, 0.0, 0.0],   # A
        [5.0, 0.0, 0.0, 0.0],   # B
        [0.5, 0.5, 0.0, 1.0],   # C (child of A)
        [5.5, 0.5, 0.0, 1.0],   # D (child of A)
    ], dtype=np.float32)

    birth_times = {'A': 0, 'B': 0, 'C': 1, 'D': 1}

    G_lin = nx.DiGraph()
    G_lin.add_edge('A', 'C')
    G_lin.add_edge('A', 'D')

    df = pd.DataFrame({
        'cell': ['A', 'B'],
        'time': [0, 0],
        'x': [0.0, 5.0],
        'y': [0.0, 0.0],
        'z': [0.0, 0.0],
        'd1': ['C', None],
        'd2': ['D', None],
    })

    return dict(
        df=df, cells=cells, idx=idx, T_max=1,
        birth_feat=birth_feat, G_lin=G_lin, birth_times=birth_times
    )


# ---------------------------------------------------------------------------
# Test 1: Shape invariant
# ---------------------------------------------------------------------------

def test_edge_attr_length_matches_edge_index():
    """
    edge_attr must have exactly one element per column of edge_index.

    Before the fix, edge_attr had len(H) elements (one per hyperedge) but
    edge_index had len(rows) columns (one per node-hyperedge membership).
    Any hyperedge with >1 node caused len(edge_attr) < edge_index.shape[1].
    """
    dataset = make_synthetic_dataset()
    snapshots = build_snapshots(dataset, k=2, spatial_radius=100.0)

    any_hyperedges_seen = False

    for t, data in enumerate(snapshots):
        if data.edge_index.numel() == 0:
            continue

        any_hyperedges_seen = True
        n_memberships = data.edge_index.shape[1]
        n_attrs = data.edge_attr.shape[0]

        assert n_memberships == n_attrs, (
            f"Snapshot t={t}: edge_index has {n_memberships} columns "
            f"but edge_attr has {n_attrs} elements.\n"
            f"edge_attr must have one entry per node-hyperedge membership, "
            f"not one per hyperedge."
        )

    assert any_hyperedges_seen, (
        "No snapshot contained any hyperedges — dataset too sparse to test."
    )


# ---------------------------------------------------------------------------
# Test 2: Type mask coverage
# ---------------------------------------------------------------------------

def test_type_masks_cover_all_memberships():
    """
    Masking edge_index by spatial (0) and lineage (1) types must collectively
    reach every column of edge_index — no membership should be silently dropped.

    Before the fix, mask indices came from e_types which only had len(H)
    elements. Only the first len(H) columns of edge_index were ever
    reachable, leaving all remaining memberships permanently unreachable
    during convolution.
    """
    dataset = make_synthetic_dataset()
    snapshots = build_snapshots(dataset, k=2, spatial_radius=100.0)

    any_hyperedges_seen = False

    for t, data in enumerate(snapshots):
        if data.edge_index.numel() == 0:
            continue

        any_hyperedges_seen = True
        total_cols = data.edge_index.shape[1]

        # Replicate exactly what the model does when filtering by type
        covered = set()
        for etype in [0, 1]:   # 0 = spatial, 1 = lineage
            mask = (data.edge_attr == etype).nonzero(as_tuple=True)[0]
            covered.update(mask.tolist())

        assert len(covered) == total_cols, (
            f"Snapshot t={t}: type masks reach only {len(covered)}/{total_cols} "
            f"edge_index columns.\n"
            f"Unreachable column indices: "
            f"{sorted(set(range(total_cols)) - covered)}\n"
            f"These memberships are silently dropped during hypergraph convolution."
        )

    assert any_hyperedges_seen, (
        "No snapshot contained any hyperedges — dataset too sparse to test."
    )
