from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.loader import DataLoader
from src.epic_dataset import EpicEmbryoDataset


def test_dataset_loader() -> None:
    print("=" * 80)
    print("Testing EpicEmbryoDataset Loader")
    print("=" * 80)
    
    processed_dir = REPO_ROOT / "Dataset" / "processed" / "by_embryo"
    manifest_path = processed_dir / "manifest.txt"
    
    if not processed_dir.exists():
        print(f"Processed directory not found: {processed_dir}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"Manifest file not found: {manifest_path}")
        sys.exit(1)
        

    print("\n Testing Local Index Mode (use_global_index=False)")
    dataset_local = EpicEmbryoDataset(processed_dir=processed_dir, manifest_path=manifest_path, use_global_index=False)
    print(f"Dataset initialized successfully with {len(dataset_local)} embryos.")
    
    sample = dataset_local[0]
    print(f"Loaded sample 0: {sample.source_file} (t0={sample.t0}, T={sample.T})")
    
    N, d, T = sample.x.shape
    print(f"  x shape:          {list(sample.x.shape)}")
    print(f"  alive_mask shape: {list(sample.alive_mask.shape)}")
    
    # Assertions
    assert sample.x.dtype == torch.float32, "x must be float32"
    assert sample.alive_mask.dtype == torch.bool, "alive_mask must be bool"
    assert sample.edge_index.dtype == torch.long, "edge_index must be long"
    assert sample.edge_t.dtype == torch.long, "edge_t must be long"
    assert sample.alive_mask.shape == (N, T), "alive_mask shape mismatch"
    assert sample.edge_index.shape[0] == 2, "edge_index must have 2 rows"
    assert len(sample.edge_t) == sample.edge_index.shape[1], "edge_t and edge_index length mismatch"
    assert len(sample.idx_to_cell) == N, "idx_to_cell length mismatch"
    
    # Verify unborn cells are zero
    unborn_mask = ~sample.alive_mask
    unborn_features = sample.x.permute(0, 2, 1)[unborn_mask]
    assert (unborn_features == 0).all(), "Found non-zero features for unborn/dead cells!"
    print("[OK] Local sample checks passed!")

    print("\n Testing Global Index Mode (use_global_index=True)")
    dataset_global = EpicEmbryoDataset(processed_dir=processed_dir, manifest_path=manifest_path, use_global_index=True)
    print(f"Dataset initialized successfully with {len(dataset_global)} embryos.")
    print(f"  Global N: {dataset_global.N_global}")
    print(f"  Global T: {dataset_global.T_global}")
    
    # Load multiple embryos to verify uniform shapes
    sample_0 = dataset_global[0]
    sample_1 = dataset_global[1]
    
    print(f"Loaded sample 0: {sample_0.source_file} (t0={sample_0.t0}, T={sample_0.T})")
    print(f"Loaded sample 1: {sample_1.source_file} (t0={sample_1.t0}, T={sample_1.T})")
    
    # Assert uniform shapes
    print(f"  sample_0 x shape: {list(sample_0.x.shape)}")
    print(f"  sample_1 x shape: {list(sample_1.x.shape)}")
    assert sample_0.x.shape == sample_1.x.shape == (dataset_global.N_global, 5, dataset_global.T_global), "Shapes must be uniform in global mode!"
    assert sample_0.alive_mask.shape == sample_1.alive_mask.shape == (dataset_global.N_global, dataset_global.T_global), "alive_mask shapes must be uniform!"
    
    # Verify unborn cells in global mode are zeroed out
    for i, sample_g in enumerate([sample_0, sample_1]):
        unborn_mask_g = ~sample_g.alive_mask
        unborn_features_g = sample_g.x.permute(0, 2, 1)[unborn_mask_g]
        assert (unborn_features_g == 0).all(), f"Found non-zero features for unborn/dead cells in global sample {i}!"
    print("[OK] Global shape uniformity and zero-masking checks passed!")
    
    # Test PyG DataLoader batching
    # Since PyG batching collates multiple Data objects into a single large disjoint graph,
    # let's verify that iterating over the loader works perfectly with batch_size=2
    dataloader = DataLoader(dataset_global, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    print(f"\nDataLoader iteration (batch_size=2) passed!")
    print(f"  Batch num_nodes:  {batch.num_nodes}")
    print(f"  Batch edge_index: {list(batch.edge_index.shape)}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_dataset_loader()
