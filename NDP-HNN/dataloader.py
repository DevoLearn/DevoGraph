# -*- coding: utf-8 -*-
"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""

#---essential imports
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset

#---Dataloader for C elegans spataial and lineage features 
class CEGrowingHypergraphDataset(Dataset):
    def __init__(self, csv_path, spatial_radius=20, max_timepoints=5, max_cells_per_tp=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['time', 'cell', 'mother', 'x', 'y', 'z'])

        self.df['cell_id'] = self.df['cell'].astype(str)
        self.df['parent_id'] = self.df['mother'].astype(str)
        self.df['time'] = pd.to_numeric(self.df['time'], errors='coerce')
        self.df[['x', 'y', 'z']] = self.df[['x', 'y', 'z']].apply(pd.to_numeric)

        all_timepoints = sorted(self.df['time'].unique())
        self.timepoints = all_timepoints[:max_timepoints]
        self.spatial_radius = spatial_radius
        self.max_cells_per_tp = max_cells_per_tp
        self.snapshots = self._generate_snapshots()

    def _generate_snapshots(self):
        snapshots = []
        for t in self.timepoints:
            frame = self.df[self.df['time'] == t].copy()
            if self.max_cells_per_tp:
                frame = frame.sample(n=min(self.max_cells_per_tp, len(frame)), random_state=42)

            frame = frame.reset_index(drop=True)
            node_map = {row['cell_id']: idx for idx, row in frame.iterrows()}
            node_features = []

            for _, row in frame.iterrows():
                lineage_depth = len(row['cell_id'])  # proxy for tree depth
                node_features.append([lineage_depth, row['x'], row['y'], row['z']])

            lineage_edges = []
            for _, row in frame.iterrows():
                if row['parent_id'] in node_map:
                    lineage_edges.append([node_map[row['parent_id']], node_map[row['cell_id']]])

            coords = frame[['x', 'y', 'z']].values
            spatial_hyperedges = []
            for i in range(len(coords)):
                group = [i]
                for j in range(len(coords)):
                    if i != j and ((coords[i] - coords[j])**2).sum()**0.5 <= self.spatial_radius:
                        group.append(j)
                if len(group) > 1:
                    spatial_hyperedges.append(list(set(group)))

            snapshots.append({
                'time': t,
                'node_features': torch.tensor(node_features, dtype=torch.float32),
                'lineage_edges': lineage_edges,
                'spatial_hyperedges': spatial_hyperedges
            })
        return snapshots

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        return self.snapshots[idx]

dataset = CEGrowingHypergraphDataset(
    "ce_temporal_data.csv",
    spatial_radius=15,
    max_timepoints=3,
    max_cells_per_tp=50
)
