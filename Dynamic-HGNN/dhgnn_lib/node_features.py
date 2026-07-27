"""Per-timepoint node feature matrices.

For every timepoint t builds a feature matrix ``X[t]`` of shape ``(N, F)`` over
the fixed cell universe (N cells), with feature columns::

    [x, y, z, size, lineage_generation, presence, vx, vy, vz, boundary_dist]

  - ``x, y, z, size``: z-score normalised over all (cell, time) pairs.
  - ``lineage_generation``: static, min-max normalised by the universe max.
  - ``presence``: 1.0 if the cell has a recorded position at time t, else 0.0.
  - ``vx, vy, vz`` (velocity): displacement of the cell since the previous
    timepoint (0 if the cell was absent then / first frame), std-normalised.
    Gives the model a *motion* signal (per-cell velocity).
  - ``boundary_dist``: distance to the nearest cell in a *different* DBSCAN
    spatial cluster at time t (0 if in noise / no other cluster), std-normalised.
    A small value means the cell sits on a community boundary and is more likely
    to switch communities.

Presence is kept at index 5 so downstream code can rely on ``X[:, 5]`` as the
present-cell mask regardless of how many extra features are appended.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

from .cell_universe import CellUniverse, lineage_generation_from_name

# DBSCAN params for the boundary-distance feature (match the spatial hyperedges).
_BND_EPS = 15.0
_BND_MIN_SAMPLES = 3


class NodeFeatureBuilder:
    """Builds and caches per-timepoint node feature tensors."""

    FEATURE_NAMES = [
        "x", "y", "z", "size", "lineage_generation", "presence",
        "vx", "vy", "vz", "boundary_dist",
    ]

    def __init__(self, raw_dir: str, universe: CellUniverse):
        self.raw_dir = raw_dir
        self.universe = universe
        self.ctd = pd.read_csv(os.path.join(raw_dir, "ce_temporal_data.csv"))
        self.timepoints = sorted(self.ctd["time"].unique().tolist())

        self._pos_mean = self.ctd[["x", "y", "z", "size"]].mean().to_numpy(dtype=np.float32)
        self._pos_std = self.ctd[["x", "y", "z", "size"]].std().to_numpy(dtype=np.float32)
        self._pos_std[self._pos_std == 0] = 1.0

        gens = np.array(
            [lineage_generation_from_name(n) for n in self.universe.names], dtype=np.float32
        )
        self._lineage_gen = gens
        self._max_gen = max(float(gens.max()), 1.0)

        self._by_time: Dict[int, pd.DataFrame] = {
            t: g.set_index("cell") for t, g in self.ctd.groupby("time")
        }

        self._precompute_dynamic_features()

    # ------------------------------------------------------------------
    # Velocity + boundary-distance precomputation
    # ------------------------------------------------------------------
    def _precompute_dynamic_features(self) -> None:
        """Precompute raw per-timepoint velocity and boundary-distance arrays."""
        N = len(self.universe)
        uni = self.universe
        # deduplicated positions per timepoint (first occurrence per cell)
        pos_by_t: Dict[int, pd.DataFrame] = {
            t: g.drop_duplicates("cell").set_index("cell")
            for t, g in self.ctd.groupby("time")
        }
        self._raw_vel: Dict[int, np.ndarray] = {}
        self._raw_bnd: Dict[int, np.ndarray] = {}

        for i, t in enumerate(self.timepoints):
            vel = np.zeros((N, 3), dtype=np.float32)
            bnd = np.zeros((N, 1), dtype=np.float32)
            cur = pos_by_t[t]

            if i > 0:
                prev = pos_by_t[self.timepoints[i - 1]]
                common = cur.index.intersection(prev.index)
                for c in common:
                    if c in uni:
                        vel[uni.get(c)] = (
                            cur.loc[c, ["x", "y", "z"]].to_numpy(np.float32)
                            - prev.loc[c, ["x", "y", "z"]].to_numpy(np.float32)
                        )

            cells = [c for c in cur.index if c in uni]
            if len(cells) >= _BND_MIN_SAMPLES:
                pts = cur.loc[cells, ["x", "y", "z"]].to_numpy(np.float64)
                lbl = DBSCAN(eps=_BND_EPS, min_samples=_BND_MIN_SAMPLES).fit_predict(pts)
                for j, c in enumerate(cells):
                    if lbl[j] == -1:
                        continue
                    other = (lbl != lbl[j]) & (lbl != -1)
                    if other.any():
                        bnd[uni.get(c), 0] = np.linalg.norm(pts[other] - pts[j], axis=1).min()

            self._raw_vel[t] = vel
            self._raw_bnd[t] = bnd

        # normalise by std over the non-zero (present / meaningful) entries
        all_vel = np.concatenate([self._raw_vel[t] for t in self.timepoints], axis=0)
        all_bnd = np.concatenate([self._raw_bnd[t] for t in self.timepoints], axis=0)
        self._vel_std = float(all_vel[all_vel != 0].std()) or 1.0
        self._bnd_std = float(all_bnd[all_bnd != 0].std()) or 1.0

    @property
    def num_features(self) -> int:
        return len(self.FEATURE_NAMES)

    def build(self, t: int) -> torch.Tensor:
        """Return the (N, 10) feature tensor for timepoint ``t``."""
        N = len(self.universe)
        feats = np.zeros((N, self.num_features), dtype=np.float32)
        feats[:, 4] = self._lineage_gen / self._max_gen

        frame = self._by_time.get(t)
        if frame is not None:
            present = frame.index.intersection(self.universe.names)
            for cell in present:
                row = frame.loc[cell]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                idx = self.universe.get(cell)
                vals = np.array([row["x"], row["y"], row["z"], row["size"]], dtype=np.float32)
                feats[idx, 0:4] = (vals - self._pos_mean) / self._pos_std
                feats[idx, 5] = 1.0

        # dynamic features (precomputed raw values, std-normalised)
        feats[:, 6:9] = self._raw_vel[t] / self._vel_std
        feats[:, 9:10] = self._raw_bnd[t] / self._bnd_std

        return torch.from_numpy(feats)

    def raw_positions(self, t: int) -> pd.DataFrame:
        """Raw (non-normalised) x,y,z,size,cell rows present at time t."""
        frame = self._by_time.get(t)
        if frame is None:
            return pd.DataFrame(columns=["cell", "x", "y", "z", "size"])
        df = frame.reset_index()
        return df[df["cell"].isin(self.universe.names)][["cell", "x", "y", "z", "size"]]
