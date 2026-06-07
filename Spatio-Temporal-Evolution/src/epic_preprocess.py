from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


DEFAULT_FEATURES: tuple[str, ...] = ("x", "y", "z", "size", "blot")


@dataclass(frozen=True)
class EpicIndex:
    cell_to_idx: dict[str, int]
    t0: int
    T: int


def _read_epic_table(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required = {"cell", "time", "x", "y", "z", "size", "blot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {file_path}")
    return df


def build_local_index(df: pd.DataFrame) -> EpicIndex:
    df = df.copy()
    df["cell"] = df["cell"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype(int)

    cells = sorted(df["cell"].unique().tolist())
    t0 = int(df["time"].min())
    t1 = int(df["time"].max())
    return EpicIndex(cell_to_idx={c: i for i, c in enumerate(cells)}, t0=t0, T=(t1 - t0) + 1)


def build_global_index(files: Iterable[str | Path]) -> EpicIndex:
    cells: set[str] = set()
    t_min: int | None = None
    t_max: int | None = None

    for fp in files:
        df = _read_epic_table(fp)
        cells.update(df["cell"].astype(str).unique().tolist())
        cur_min = int(pd.to_numeric(df["time"], errors="coerce").min())
        cur_max = int(pd.to_numeric(df["time"], errors="coerce").max())
        t_min = cur_min if t_min is None else min(t_min, cur_min)
        t_max = cur_max if t_max is None else max(t_max, cur_max)

    if t_min is None or t_max is None:
        raise ValueError("No files / empty time ranges while building global index.")

    cell_list = sorted(cells)
    cell_to_idx = {c: i for i, c in enumerate(cell_list)}
    T = (t_max - t_min) + 1
    return EpicIndex(cell_to_idx=cell_to_idx, t0=t_min, T=T)


def preprocess_epic_file_sparse(
    file_path: str | Path,
    *,
    distance_threshold: float = 20.0,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, EpicIndex]:
    """
    Returns:
      X: (N, d, T) float32
      alive_mask: (N, T) bool
      edge_src: (E,) int32
      edge_dst: (E,) int32
      edge_t: (E,) int32  (0-indexed time)
      index: local EpicIndex (per-file)
    """
    df = _read_epic_table(file_path)
    index = build_local_index(df)
    df = df.copy()
    df["cell"] = df["cell"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype(int)

    N = len(index.cell_to_idx)
    T = index.T
    d = len(features)

    X = np.zeros((N, d, T), dtype=np.float32)
    alive_mask = np.zeros((N, T), dtype=bool)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_t: list[int] = []

    # Populate X and alive_mask
    # NOTE: multiple rows for same (cell,time) should not happen; if it does, last row wins.
    for _, row in df.iterrows():
        cell = row["cell"]
        if cell not in index.cell_to_idx:
            continue
        t_idx = int(row["time"] - index.t0)
        if t_idx < 0 or t_idx >= T:
            continue
        c_idx = index.cell_to_idx[cell]
        X[c_idx, :, t_idx] = [float(row[f]) for f in features]
        alive_mask[c_idx, t_idx] = True

    # Build edges per time step using only alive cells at that time.
    for t_idx in range(T):
        t_val = index.t0 + t_idx
        alive_df = df[df["time"] == t_val]
        if alive_df.empty:
            continue

        alive_cells = alive_df["cell"].tolist()
        alive_indices = [index.cell_to_idx[c] for c in alive_cells if c in index.cell_to_idx]
        if len(alive_indices) == 0:
            continue

        # Spatial undirected edges based on (x,y,z) proximity.
        coords = alive_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
        if coords.shape[0] >= 2:
            distances = squareform(pdist(coords, metric="euclidean"))
            close = (distances < distance_threshold) & (distances > 0)
            for i, idx_i in enumerate(alive_indices):
                js = np.nonzero(close[i])[0]
                for j in js.tolist():
                    idx_j = alive_indices[j]
                    edge_src.append(idx_i)
                    edge_dst.append(idx_j)
                    edge_t.append(t_idx)
                    edge_src.append(idx_j)
                    edge_dst.append(idx_i)
                    edge_t.append(t_idx)

    
        for cell in alive_cells:
            if not cell:
                continue
            last = cell[-1]
            if not last.isalpha():
                continue
            parent = cell[:-1]
            if parent in index.cell_to_idx and cell in index.cell_to_idx:
                p_idx = index.cell_to_idx[parent]
                c_idx = index.cell_to_idx[cell]
                edge_src.append(p_idx)
                edge_dst.append(c_idx)
                edge_t.append(t_idx)

    return (
        X,
        alive_mask,
        np.asarray(edge_src, dtype=np.int32),
        np.asarray(edge_dst, dtype=np.int32),
        np.asarray(edge_t, dtype=np.int32),
        index,
    )

