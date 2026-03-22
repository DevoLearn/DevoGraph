"""Unified graph builder for C. elegans embryogenesis data.

This is the central module of the unified D-GNN framework.  It takes a single
CSV of cell-lineage data and can produce **both** graph representations used
in the existing codebase:

  1. **Snapshot sequence** (NDP-HNN style) — a list of PyG ``Data`` objects,
     one per discrete time step, each containing alive nodes, spatial + lineage
     hyperedges, and per-edge-type attributes.

  2. **Continuous-time dynamic graph** (DevoTG style) — a single PyG
     ``TemporalData`` object with event-based parent→daughter edges,
     timestamps, and padded feature vectors.

Both representations share a **common node registry** (cell→ID mapping,
birth times, lineage graph) computed once at load time.  This means embeddings
or analyses produced by either branch are directly comparable — which is the
prerequisite for the planned unification of NDP-HNN and DevoTG into one model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, TemporalData
from torch_geometric.utils import from_scipy_sparse_matrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_hyperedges_alive(
    t: int,
    cells: List[str],
    idx: Dict[str, int],
    G_lin: nx.DiGraph,
    birth_feat: np.ndarray,
    birth_times: Dict[str, int],
    k: int = 5,
    spatial_r: float = 25.0,
) -> Tuple[List[int], List[Tuple[int, ...]], Dict[Tuple[int, ...], str]]:
    """Return (alive_idx, hyperedge_list, he→type_tag) at time *t*."""

    alive = [c for c in cells if c in birth_times and birth_times[c] <= t]
    alive_idx = [idx[c] for c in alive]

    # ── spatial hyperedges (KNN) ──────────────────────────────────────
    spatial: List[Tuple[int, ...]] = []
    if len(alive_idx) >= 2:
        X = birth_feat[alive_idx, :3]
        nbrs = NearestNeighbors(
            n_neighbors=min(k + 1, len(alive_idx)), radius=spatial_r
        ).fit(X)
        _, knn = nbrs.kneighbors(X)
        for row in knn:
            he = tuple(sorted({alive_idx[i] for i in row}))
            if len(he) > 1:
                spatial.append(he)

    # ── lineage hyperedges (siblings alive at t) ──────────────────────
    lineage: List[Tuple[int, ...]] = []
    for p in alive:
        kids = [
            child
            for child in G_lin.successors(p)
            if child in birth_times and birth_times[child] <= t
        ]
        if len(kids) > 1:
            lineage.append(tuple(sorted(idx[k_] for k_ in kids)))

    # ── deduplicate & tag ─────────────────────────────────────────────
    he2type: Dict[Tuple[int, ...], str] = {}
    for he in spatial:
        he2type.setdefault(he, "spatial")
    for he in lineage:
        he2type.setdefault(he, "lineage")

    seen: set = set()
    unique: List[Tuple[int, ...]] = []
    for he in list(spatial) + list(lineage):
        if he not in seen:
            unique.append(he)
            seen.add(he)

    return alive_idx, unique, he2type


def _pad_feature(base: List[float], dim: int = 172) -> torch.Tensor:
    """Left-align *base* into a zero-padded tensor of length *dim*."""
    out = torch.zeros(dim, dtype=torch.float32)
    out[: len(base)] = torch.tensor(base, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class UnifiedGraphBuilder:
    """Build snapshot and/or CTDG representations from one CSV.

    The key design decision is to compute the **shared node registry** —
    cell list, ID map, birth times, lineage DAG, birth features — exactly once
    in ``__init__``, so that both ``build_snapshots`` and ``build_ctdg`` operate
    on the same data and produce compatible node IDs.

    Parameters
    ----------
    csv_path : str
        Path to the cell-lineage CSV (columns: Parent Cell, Birth Time,
        parent_x/y/z, Daughter 1, Daughter 2).
    knn_k : int
        Number of nearest neighbours for spatial hyperedges.
    spatial_radius : float
        Distance threshold for spatial hyperedges.
    ctdg_feature_dim : int
        Feature-vector length for the CTDG representation (padded).
    """

    def __init__(
        self,
        csv_path: str,
        knn_k: int = 5,
        spatial_radius: float = 25.0,
        ctdg_feature_dim: int = 172,
    ):
        self.csv_path = csv_path
        self.knn_k = knn_k
        self.spatial_radius = spatial_radius
        self.ctdg_feature_dim = ctdg_feature_dim

        # Load once, share everywhere.
        self._raw = self._load_csv(csv_path)

    # ------------------------------------------------------------------
    # Shared data loading (called once)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(csv_path: str) -> Dict[str, Any]:
        """Parse the lineage CSV into the shared node registry.

        Returns a dict with keys:
            df, cells, idx, T_max, birth_feat, G_lin, birth_times
        """
        df = pd.read_csv(csv_path)
        df = df.rename(
            columns={
                "Parent Cell": "cell",
                "Birth Time": "time",
                "parent_x": "x",
                "parent_y": "y",
                "parent_z": "z",
                "Daughter 1": "d1",
                "Daughter 2": "d2",
            }
        )
        for col in ("cell", "d1", "d2"):
            if col in df.columns:
                df[col] = df[col].astype(str)

        cells = sorted(set(df["cell"]) | set(df["d1"]) | set(df["d2"]))
        if "nan" in cells:
            cells.remove("nan")
        idx = {c: i for i, c in enumerate(cells)}

        T_max = int(df["time"].max())

        # Birth features: [x, y, z, t/T_max]
        birth_feat = np.zeros((len(cells), 4), dtype=np.float32)
        for _, r in df.iterrows():
            i = idx[r.cell]
            birth_feat[i, :3] = [r.x, r.y, r.z]
            birth_feat[i, 3] = r.time / T_max
            for child in (r.d1, r.d2):
                if pd.notna(child) and child != "nan":
                    j = idx[child]
                    birth_feat[j, :3] = [r.x, r.y, r.z]
                    birth_feat[j, 3] = r.time / T_max

        # Directed lineage graph
        G_lin = nx.DiGraph()
        for _, r in df.iterrows():
            if pd.notna(r.d1) and r.d1 != "nan":
                G_lin.add_edge(r.cell, r.d1)
            if pd.notna(r.d2) and r.d2 != "nan":
                G_lin.add_edge(r.cell, r.d2)

        birth_times: Dict[str, int] = {}
        for _, r in df.iterrows():
            birth_times[r.cell] = int(r.time)
            for child in (r.d1, r.d2):
                if pd.notna(child) and child != "nan":
                    birth_times[child] = int(r.time)

        return dict(
            df=df,
            cells=cells,
            idx=idx,
            T_max=T_max,
            birth_feat=birth_feat,
            G_lin=G_lin,
            birth_times=birth_times,
        )

    # ------------------------------------------------------------------
    # Accessors for shared state
    # ------------------------------------------------------------------

    @property
    def cells(self) -> List[str]:
        return self._raw["cells"]

    @property
    def idx(self) -> Dict[str, int]:
        return self._raw["idx"]

    @property
    def num_cells(self) -> int:
        return len(self._raw["cells"])

    @property
    def T_max(self) -> int:
        return self._raw["T_max"]

    @property
    def birth_feat(self) -> np.ndarray:
        return self._raw["birth_feat"]

    @property
    def birth_times(self) -> Dict[str, int]:
        return self._raw["birth_times"]

    @property
    def lineage_graph(self) -> nx.DiGraph:
        return self._raw["G_lin"]

    # ------------------------------------------------------------------
    # Branch 1: snapshot sequence (NDP-HNN style)
    # ------------------------------------------------------------------

    def build_snapshots(self) -> List[Data]:
        """Build one PyG ``Data`` per time step with hyperedge incidence.

        Each snapshot contains *all* cells as nodes (with birth features),
        spatial + lineage hyperedges among alive cells, and an ``edge_attr``
        tensor encoding hyperedge type (0 = spatial, 1 = lineage).
        """
        d = self._raw
        cells, idx, G_lin = d["cells"], d["idx"], d["G_lin"]
        birth_feat, birth_times = d["birth_feat"], d["birth_times"]
        T_max = d["T_max"]

        snapshots: List[Data] = []
        for t in range(T_max + 1):
            alive_idx, H, he2type = _make_hyperedges_alive(
                t, cells, idx, G_lin, birth_feat, birth_times,
                k=self.knn_k, spatial_r=self.spatial_radius,
            )

            rows, cols = [], []
            for e_id, nodes in enumerate(H):
                rows.extend(nodes)
                cols.extend([e_id] * len(nodes))

            if len(rows) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                e_types = torch.empty((0,), dtype=torch.long)
            else:
                Hmat = coo_matrix(
                    (np.ones(len(rows)), (rows, cols)),
                    shape=(len(cells), len(H)),
                )
                edge_index, _ = from_scipy_sparse_matrix(Hmat)
                he_type_list = [
                    0 if he2type[tuple(sorted(H[e]))] == "spatial" else 1
                    for e in range(len(H))
                ]
                e_types = torch.tensor(
                    [he_type_list[c] for c in cols], dtype=torch.long
                )

            data = Data(
                x=torch.tensor(birth_feat, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=e_types,
                t=torch.tensor([t] * len(cells), dtype=torch.long),
            )
            snapshots.append(data)

        return snapshots

    # ------------------------------------------------------------------
    # Branch 2: continuous-time dynamic graph (DevoTG style)
    # ------------------------------------------------------------------

    def build_ctdg(self) -> TemporalData:
        """Build a ``TemporalData`` object with parent→daughter events.

        Uses the **same node IDs** as ``build_snapshots`` so that embeddings
        from both branches are directly comparable.
        """
        df, idx = self._raw["df"], self._raw["idx"]
        birth_feat = self._raw["birth_feat"]
        dim = self.ctdg_feature_dim

        src_list, dst_list, t_list, msg_list = [], [], [], []

        for _, r in df.iterrows():
            p_id = idx[r.cell]
            for child_col in (r.d1, r.d2):
                if pd.notna(child_col) and child_col != "nan":
                    c_id = idx[child_col]
                    src_list.append(p_id)
                    dst_list.append(c_id)
                    t_list.append(float(r.time))
                    msg = _pad_feature(
                        [r.x, r.y, r.z, 0.0, 0.0, 0.0, float(r.time)], dim
                    )
                    msg_list.append(msg)

        # Node features: pad the 4-dim birth features to ctdg_feature_dim
        node_features = torch.zeros(len(self.cells), dim, dtype=torch.float32)
        node_features[:, :4] = torch.tensor(birth_feat, dtype=torch.float32)

        data = TemporalData(
            src=torch.tensor(src_list, dtype=torch.long),
            dst=torch.tensor(dst_list, dtype=torch.long),
            t=torch.tensor(t_list, dtype=torch.long),
            msg=torch.stack(msg_list) if msg_list else torch.empty(0, dim),
            x=node_features,
        )
        return data

    # ------------------------------------------------------------------
    # Convenience: build both
    # ------------------------------------------------------------------

    def build(self, mode: str = "both") -> Dict[str, Any]:
        """Build graph representations according to *mode*.

        Parameters
        ----------
        mode : str
            ``"snapshot"`` — only snapshot sequence.
            ``"ctdg"``     — only continuous-time dynamic graph.
            ``"both"``     — both (default).

        Returns
        -------
        dict with keys ``"snapshots"`` and/or ``"ctdg"``.
        """
        result: Dict[str, Any] = {}
        if mode in ("snapshot", "both"):
            result["snapshots"] = self.build_snapshots()
        if mode in ("ctdg", "both"):
            result["ctdg"] = self.build_ctdg()
        return result
