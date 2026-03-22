# Unified D-GNN PoC

Proof-of-concept for unifying the two existing DevoGraph approaches into a
single pipeline:

- **NDP-HNN** (snapshot-based hypergraph temporal GNN)
- **DevoTG** (continuous-time dynamic graph with TGN memory)

## What this demonstrates

The central problem: NDP-HNN and DevoTG process the same biological data
(*C. elegans* cell lineage) but produce **incompatible** graph representations
with different node IDs, feature formats, and temporal semantics.

This PoC shows the unification approach: a `UnifiedGraphBuilder` that loads the
lineage CSV once and builds **both** representations with a shared node registry
(same cell-to-ID mapping, same birth features). This makes embeddings from
either branch directly comparable.

It then runs a minimal training loop on the snapshot branch (HypergraphConv +
GRU) to verify the model learns (loss decreases).

## Files

| File | Purpose |
|---|---|
| `config.py` | Unified dataclass config (merges NDP-HNN + DevoTG settings) |
| `graph_builder.py` | **Core**: builds snapshot sequence AND CTDG from one CSV |
| `model.py` | DevoGNN model (snapshot branch functional, CTDG branch stubbed) |
| `run_poc.py` | End-to-end demo script |

## How to run

From the repository root:

```bash
# Install dependencies (if not already installed)
pip install torch torch-geometric numpy pandas scipy scikit-learn networkx

# Run the PoC (defaults: 5 epochs, both graph modes)
python3 -m devograph.unified.run_poc

# Customize
python3 -m devograph.unified.run_poc --epochs 10 --mode snapshot
python3 -m devograph.unified.run_poc --mode ctdg  # builds CTDG only, skips training
```

Expected output:

```
Loading data from .../cells_birth_and_pos.csv
  Cells: 1203  |  T_max: 735
  Snapshots: 736 time steps  |  ...
  CTDG: 1203 nodes  |  1284 division events  |  Feature dim: 172
  Node IDs consistent across both branches.

Training DevoGNN (hgcn/gru) for 5 epochs on cpu
  Epoch 001 — avg loss: 50662.72
  Epoch 002 — avg loss: 36098.86
  ...
PoC complete.
```

## What comes next

This PoC lays the groundwork for the full GSoC project:

1. **CTDG training branch** — wire up TGN memory + link prediction using the
   same `UnifiedGraphBuilder` output
2. **Multiscale hypergraph** — add DBSCAN-based tissue-level hyperedges and
   learned pooling layers
3. **Dynamic state expansion** — replace the fixed-size hidden state with
   parent→daughter state inheritance for growing graphs
4. **Visualization dashboard** — temporal graph playback and embedding explorer
