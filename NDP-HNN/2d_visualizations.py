# -*- coding: utf-8 -*-
"""HNNs C Elegans Embryogensis

Contributer: Lalith Bharadwaj Baru
"""

#---essential imports
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from dataloader import CEGrowingHypergraphDataset

#---load the temporal data 
dataset = CEGrowingHypergraphDataset("./data/ce_temporal_data.csv", max_timepoints=3, max_cells_per_tp=50)
snapshot = dataset[0]

node_feats = snapshot['node_features']
lineage_edges = snapshot['lineage_edges']

# Create PyG Data object
if len(lineage_edges) > 0:
    edge_index = torch.tensor(lineage_edges, dtype=torch.long).t().contiguous()
else:
    edge_index = torch.empty((2, 0), dtype=torch.long)

pyg_data = Data(x=node_feats, edge_index=edge_index)

#---2d networkx visualization
G = nx.DiGraph()
for i in range(node_feats.shape[0]):
    G.add_node(i, pos=(node_feats[i][1].item(), node_feats[i][2].item()))  # use x, y only

for edge in lineage_edges:
    G.add_edge(edge[0], edge[1])

plt.figure(figsize=(6, 6))
nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=True,
        node_color='lightblue', node_size=300, arrows=True)
plt.title(f"Lineage Structure at Time {snapshot['time']}")
plt.show()