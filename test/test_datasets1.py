import sys
import types
from unittest.mock import MagicMock

# hydra is not installed in this env; mock it before importing the module
hydra_mock = types.ModuleType("hydra")
hydra_utils_mock = types.ModuleType("hydra.utils")
hydra_utils_mock.get_original_cwd = MagicMock(return_value="/tmp")
hydra_mock.utils = hydra_utils_mock
sys.modules.setdefault("hydra", hydra_mock)
sys.modules.setdefault("hydra.utils", hydra_utils_mock)

import numpy as np
import torch

from devograph.datasets.datasets1 import CellTrackDataset


def make_dataset():
    """Bypass __init__ and set only the attributes edge_feat_embedding needs."""
    ds = object.__new__(CellTrackDataset)
    ds.edge_feat_embed_dict = {'p': 1, 'use_normalized_x': True, 'normalized_features': True}
    ds.which_preprocess = 'MinMax'
    ds.separate_models = False
    ds.normalize_cols = np.array([True, True, True])
    return ds


def test_edge_feat_embedding_empty_edge_index():
    """No crash and correct shape when graph has zero edges."""
    ds = make_dataset()
    x = np.random.rand(5, 3).astype(np.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    result = ds.edge_feat_embedding(x, edge_index)

    assert result.shape == (0, 3), f"Expected (0, 3), got {result.shape}"


def test_edge_feat_embedding_nonempty_edge_index():
    """Normal case: returns (E, F) edge features."""
    ds = make_dataset()
    x = np.random.rand(5, 3).astype(np.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    result = ds.edge_feat_embedding(x, edge_index)

    assert result.shape == (3, 3), f"Expected (3, 3), got {result.shape}"
