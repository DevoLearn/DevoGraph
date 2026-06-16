from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def initialize_biological_graph(
    df_active: pd.DataFrame,
    cell_to_idx: dict[str, int],
    N: int,
    *,
    distance_threshold: float = 20.0,
) -> np.ndarray:
    """
    Build the initial adjacency matrix A^t for a single time step.

    Combines:
      1. Spatial proximity edges (undirected) from 3D Euclidean distance.
      2. Lineage edges (directed parent -> daughter) from C. elegans naming.

    Args:
        df_active: Rows for cells alive at time t (must include cell, x, y, z).
        cell_to_idx: Global cell-name to node-index mapping for this embryo.
        N: Total number of nodes in the graph (including currently inactive cells).
        distance_threshold: Max 3D distance for spatial adjacency (default 20.0).

    Returns:
        A_t: Dense adjacency matrix of shape (N, N), dtype float32.
    """
    if df_active.empty:
        return np.zeros((N, N), dtype=np.float32)

    A_t = np.zeros((N, N), dtype=np.float32)
    active_cells = df_active["cell"].astype(str).values
    active_indices = [cell_to_idx[c] for c in active_cells if c in cell_to_idx]
    if not active_indices:
        return A_t

    coords = df_active[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
    if coords.shape[0] >= 2:
        distances = squareform(pdist(coords, metric="euclidean"))
        close = (distances < distance_threshold) & (distances > 0)
        for i, idx_i in enumerate(active_indices):
            for j in np.nonzero(close[i])[0]:
                idx_j = active_indices[j]
                A_t[idx_i, idx_j] = 1.0

    for cell in active_cells:
        if not cell:
            continue
        last = cell[-1]
        if not last.isalpha():
            continue
        parent = cell[:-1]
        if parent in cell_to_idx and cell in cell_to_idx:
            p_idx = cell_to_idx[parent]
            c_idx = cell_to_idx[cell]
            A_t[p_idx, c_idx] = 1.0

    return A_t


def adjacency_from_sparse(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_t: np.ndarray,
    t_idx: int,
    N: int,
) -> np.ndarray:
    """
    Reconstruct dense A^t from preprocessed sparse edge lists at timestep t_idx.
    """
    A_t = np.zeros((N, N), dtype=np.float32)
    mask = edge_t == t_idx
    if not np.any(mask):
        return A_t
    src = edge_src[mask]
    dst = edge_dst[mask]
    A_t[src, dst] = 1.0
    return A_t


def node_importance_scores(A_t: np.ndarray) -> np.ndarray:
    """
    ESTGEL node importance: total degree (in-degree + out-degree) at time t.
    """
    return A_t.sum(axis=0) + A_t.sum(axis=1)


def generate_nested_subgraphs(A_t: np.ndarray, K: int) -> np.ndarray:
    """
    Decompose A^t into K nested subgraphs following ESTGEL EAM decomposition.

    Produces G^t_0 ⊇ G^t_1 ⊇ ... ⊇ G^t_K by iteratively removing the S lowest-
    importance nodes and all edges incident to them, where S = floor(N / K).

    Args:
        A_t: Initial adjacency matrix, shape (N, N).
        K: Number of decomposition iterations (output has K + 1 layers).

    Returns:
        Nested subgraph tensor of shape (N, N, K + 1), matching the paper layout.
    """
    if K < 1:
        raise ValueError("K must be >= 1 for nested subgraph decomposition.")

    N = A_t.shape[0]
    if A_t.shape[1] != N:
        raise ValueError(f"A_t must be square; got shape {A_t.shape}.")

    S = N // K
    if S < 1:
        raise ValueError(f"S = N // K must be >= 1; got N={N}, K={K}, S={S}.")

    subgraphs = np.zeros((K + 1, N, N), dtype=np.float32)
    subgraphs[0] = A_t.astype(np.float32, copy=False)

    current_A = A_t.astype(np.float32, copy=True)
    importance = node_importance_scores(current_A)
    active_nodes = np.ones(N, dtype=bool)

    for k in range(1, K + 1):
        masked_scores = np.where(active_nodes, importance, np.inf)
        lowest_nodes = np.argsort(masked_scores)[:S]
        active_nodes[lowest_nodes] = False

        next_A = current_A.copy()
        removed = np.where(~active_nodes)[0]
        next_A[removed, :] = 0.0
        next_A[:, removed] = 0.0

        subgraphs[k] = next_A
        current_A = next_A

    return np.transpose(subgraphs, (1, 2, 0))


def build_nested_subgraph_tensor(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_t: np.ndarray,
    t_idx: int,
    N: int,
    K: int,
) -> np.ndarray:
    """
    End-to-end helper: sparse edges at time t -> nested subgraph tensor (N, N, K+1).
    """
    A_t = adjacency_from_sparse(edge_src, edge_dst, edge_t, t_idx, N)
    return generate_nested_subgraphs(A_t, K)
