"""Shared library code for the DHGNN notebooks.

Notebooks under ``notebooks/`` add the project root to ``sys.path`` and do::

    from dhgnn_lib.cell_universe import CellUniverse
    from dhgnn_lib import hyperedges, incidence, labels, node_features, visualization, training
    from dhgnn_lib.hypersagnn import HyperSAGNNLayer
    from dhgnn_lib.dhgnn_model import DHGNNModel, DynamicHypergraphStack

Modules:
    cell_universe   - fixed cell name <-> index universe (N = 1625 cells)
    node_features   - per-timepoint (N, 10) node feature matrices
    hyperedges      - builders for spatial/lineage/functional hyperedges
                       + writers for the lineage/spatial lookup tables
    incidence       - HyperedgeSet / HyperedgeBatch + H_aug(t) assembly
    labels          - fate label tensor
    sparse_ops      - scatter_sum/mean/max/softmax (pure PyTorch)
    hypersagnn      - Hyper-SAGNN V2E2V convolution layer
    dhgnn_model     - DynamicHypergraphStack + DHGNNModel (fate head)
    training        - fate loss, metrics, optimizer/scheduler, train loop
    visualization   - plotting helpers for all notebooks

Keeping the heavy lifting here keeps the notebooks focused on
exploration / visualization / narrative, while the underlying logic stays
unit-testable and reusable across notebooks.
"""
