"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from hyperedges import make_hyperedges_alive

def build_snapshots(dataset: Dict[str, Any],
                    k: int,
                    spatial_radius: float) -> List[Data]:
    df = dataset['df']; cells = dataset['cells']
    idx = dataset['idx']; T_max = dataset['T_max']
    birth_feat = dataset['birth_feat']; G_lin = dataset['G_lin']
    birth_times = dataset['birth_times']

    snapshots = []
    for t in range(int(T_max) + 1):
        alive_idx, H, he2type = make_hyperedges_alive(
            t, cells, df, idx, G_lin, birth_feat, birth_times,
            k=k, spatial_r=spatial_radius
        )
        #--- incidence -> bipartite edge_index
        rows, cols = [], []
        for e_id, nodes in enumerate(H):
            rows.extend(nodes)
            cols.extend([e_id] * len(nodes))
        if len(rows) == 0:
            Hmat = coo_matrix((0,0))
            edge_index = torch.empty((2,0), dtype=torch.long)
            e_types = torch.empty((0,), dtype=torch.long)
        else:
            Hmat = coo_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(len(cells), len(H)))
            edge_index, _ = from_scipy_sparse_matrix(Hmat)
            #--- [0=spatial, 1=lineage]
            e_types = torch.tensor(
                [0 if he2type[tuple(sorted(H[e]))] == 'spatial' else 1
                 for e in range(len(H))],
                dtype=torch.long
            )

        data = Data(
            x=torch.tensor(birth_feat, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=e_types,
            t=torch.tensor([t] * len(cells), dtype=torch.long)
        )
        snapshots.append(data)
    return snapshots
