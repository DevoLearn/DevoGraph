"""HNNs C Elegans Embryogenesis

Contributor: Lalith Bharadwaj Baru
"""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    #--- data
    csv_path: str = "cells_birth_and_pos.csv"

    #--- hyperedges
    knn_k: int = 5
    spatial_radius: float = 25.0

    #--- model
    in_dim: int = 4
    hid_dim: int = 64
    out_dim: int = 64
    num_edge_types: int = 2    # 0=spatial, 1=lineage
    conv_type: str = "hgcn"    # ["hgcn","hsage","ugnn"]
    rnn_type: str = "gru"      # ["gru","lstm","rnn"]
    use_transformer: bool = False

    #--- training
    epochs: int = 30
    lr: float = 1e-3
    seed: int = 42

    #--- io
    save_dir: str = "outputs"
    embeddings_path: str = field(init=False)

    def __post_init__(self):
        """Automatically generate embedding filename based on conv and rnn type."""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.embeddings_path = str(
            Path(self.save_dir) / f"embeddings_{self.conv_type}_{self.rnn_type}.npy"
        )
