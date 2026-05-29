# Spatio-Temporal-Evolution
This model tracks two critical dimensions of growth simultaneously: the 3D spatial arrangement of cells and the directed cell lineage. By combining these spatial and lineage pathways with an "edge attention mechanism", the tool highlights the most critical cell-to-cell connections at various stages of growth.

## EPIC preprocessing (build X and A tensors)
This repo includes a dataset having the converted EPIC single-file format in `dataset/raw/*.csv` into:

- **X**: `(N, d, T)` where `d=5` for `["x","y","z","size","blot"]`
- **Edges (sparse A)**: `edge_src, edge_dst, edge_t` representing \(A_t[i,j]=1\) (spatial proximity edges + directed lineage edges)
- **alive_mask**: `(N, T)` boolean mask for “growing node” handling
