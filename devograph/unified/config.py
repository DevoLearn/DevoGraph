"""Unified configuration for the D-GNN pipeline.

Merges NDP-HNN Config and DevoTG config.yaml into a single dataclass
that controls data loading, graph construction, and model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UnifiedConfig:
    # ── data ──────────────────────────────────────────────────────────
    csv_path: str = "NDP-HNN/data/cells_birth_and_pos.csv"

    # ── graph construction ────────────────────────────────────────────
    graph_mode: str = "snapshot"  # "snapshot" | "ctdg" | "both"
    knn_k: int = 5
    spatial_radius: float = 25.0
    ctdg_feature_dim: int = 172

    # ── model ─────────────────────────────────────────────────────────
    in_dim: int = 4               # node feature dim for snapshot mode
    hid_dim: int = 64
    out_dim: int = 64
    num_edge_types: int = 2       # 0=spatial, 1=lineage
    conv_type: str = "hgcn"       # "hgcn" | "hsage" | "ugnn"
    rnn_type: str = "gru"         # "gru" | "lstm" | "rnn"

    # ── training ──────────────────────────────────────────────────────
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42

    # ── io ────────────────────────────────────────────────────────────
    save_dir: str = "outputs/unified"

    def __post_init__(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
