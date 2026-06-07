from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data

DEFAULT_FEATURES: tuple[str, ...] = ("x", "y", "z", "size", "blot")

class EpicEmbryoDataset(Dataset):
    """
    PyTorch Dataset wrapper for processed EPIC embryo developmental trajectories.
    Each sample represents an embryo sequence loaded from compressed NPZ format.
    Yields PyTorch Geometric `Data` objects containing dynamic graph sequences.
    
    If use_global_index is True, it dynamically scales/aligns all sample tensors
    to uniform shapes (N_global, d, T_global) on-the-fly.
    """
    
    def __init__(
        self,
        processed_dir: str | Path,
        manifest_path: str | Path | None = None,
        features: tuple[str, ...] = DEFAULT_FEATURES,
        transform=None,
        use_global_index: bool = False,
    ) -> None:
        """
        Args:
            processed_dir: Path to directory containing processed .npz files.
            manifest_path: Optional path to manifest.txt. If None, defaults to processed_dir / manifest.txt.
            features: Tuple of node feature names to expect.
            transform: Optional transform to apply to PyG Data object.
            use_global_index: Whether to pad/mask data to global dimensions on-the-fly.
        """
        self.processed_dir = Path(processed_dir)
        self.features = features
        self.transform = transform
        self.use_global_index = use_global_index
        
        if manifest_path is None:
            self.manifest_path = self.processed_dir / "manifest.txt"
        else:
            self.manifest_path = Path(manifest_path)
            
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
            
        # Read file list from manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.filenames = [line.strip() for line in f if line.strip()]
            
        if self.use_global_index:
            self._build_global_index()
            
    def _build_global_index(self) -> None:
        """
        Parses all NPZ files listed in the manifest once at initialization,
        determining the global unique cells and maximum time span.
        """
        cells: set[str] = set()
        T_max = 0
        
        for name in self.filenames:
            filepath = self.processed_dir / name
            npz = np.load(filepath, allow_pickle=True)
            cells.update(npz["idx_to_cell"])
            T_max = max(T_max, int(npz["T"]))
            
        self.global_cell_list = sorted(cells)
        self.global_cell_to_idx = {c: i for i, c in enumerate(self.global_cell_list)}
        self.N_global = len(self.global_cell_list)
        self.T_global = T_max
        
    def __len__(self) -> int:
        return len(self.filenames)
        
    def __getitem__(self, idx: int) -> Data:
        """
        Loads the idx-th embryo dataset file and wraps it in a PyG Data object.
        
        Returns:
            A torch_geometric.data.Data object with attributes:
                x: (N, d, T) float32 node features
                alive_mask: (N, T) bool mask indicating if cell is active
                edge_index: (2, E) long edge connections
                edge_t: (E,) long edge timestamps
                idx_to_cell: (N,) object array of cell name strings
                t0: int start time offset
                T: int total sequence steps
                source_file: string filename
        """
        filepath = self.processed_dir / self.filenames[idx]
        npz = np.load(filepath, allow_pickle=True)
        
        X_local = torch.tensor(npz["X"], dtype=torch.float32)
        alive_mask_local = torch.tensor(npz["alive_mask"], dtype=torch.bool)
        
        edge_src = npz["edge_src"]
        edge_dst = npz["edge_dst"]
        edge_t = torch.tensor(npz["edge_t"], dtype=torch.long)
        
        idx_to_cell = npz["idx_to_cell"]
        t0 = int(npz["t0"])
        T_local = int(npz["T"])
        source_file = str(npz["source_file"])
        
        if self.use_global_index:
            # Map local indices to global index locations
            global_indices = [self.global_cell_to_idx[cell] for cell in idx_to_cell]
            global_indices_tensor = torch.tensor(global_indices, dtype=torch.long)
            
            # 1. Initialize global tensors as zeros
            N_global = self.N_global
            T_global = self.T_global
            d = len(self.features)
            
            X = torch.zeros((N_global, d, T_global), dtype=torch.float32)
            alive_mask = torch.zeros((N_global, T_global), dtype=torch.bool)
            
            # 2. Populate/unmask active cell features
            X[global_indices_tensor, :, :T_local] = X_local
            alive_mask[global_indices_tensor, :T_local] = alive_mask_local
            
            # 3. Map edge sources and destinations to global indices
            if len(edge_src) > 0:
                edge_index = torch.stack([
                    global_indices_tensor[torch.tensor(edge_src, dtype=torch.long)],
                    global_indices_tensor[torch.tensor(edge_dst, dtype=torch.long)]
                ], dim=0)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            idx_to_cell = np.array(self.global_cell_list, dtype=object)
            T = T_global
        else:
            X = X_local
            alive_mask = alive_mask_local
            if len(edge_src) > 0:
                edge_index = torch.stack([
                    torch.tensor(edge_src, dtype=torch.long),
                    torch.tensor(edge_dst, dtype=torch.long)
                ], dim=0)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            T = T_local
            
        data = Data(
            x=X,
            alive_mask=alive_mask,
            edge_index=edge_index,
            edge_t=edge_t,
            idx_to_cell=idx_to_cell,
            t0=t0,
            T=T,
            source_file=source_file
        )
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data
