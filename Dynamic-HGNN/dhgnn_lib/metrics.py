"""Evaluation metrics for the DHGNN.

Two headline metrics, both computed on the trained model (never trained on):

  - ``fate_accuracy``: supervised cell-fate accuracy on a set of cells.
  - ``lineage_coherence``: representation-quality of the learned embeddings --
    do same-lineage cells cluster together? (silhouette + k-NN purity).
"""

from __future__ import annotations

import numpy as np
import torch


def fate_accuracy(fate_logits: torch.Tensor, fate: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of masked cells whose predicted founder lineage is correct."""
    m = mask.bool()
    if m.sum() == 0:
        return 0.0
    pred = fate_logits[m].argmax(dim=-1)
    return (pred == fate[m]).float().mean().item()


def lineage_coherence(embeddings, labels, k: int = 10):
    """Do same-lineage cells cluster together in embedding space?

    Clean, non-leaky representation-quality metric (lineage comes from the cell
    name, never a target). Returns ``(silhouette, knn_purity)``:

    - ``silhouette`` in [-1, 1]: higher = tighter, better-separated lineage
      clusters (positive = genuine clusters, negative = jumbled).
    - ``knn_purity`` in [0, 1]: fraction of each cell's ``k`` nearest neighbours
      (in embedding space) that share its lineage; higher = more coherent.

    ``labels`` are the per-cell lineage/founder class indices aligned with
    ``embeddings``. Returns ``(nan, nan)`` if fewer than 2 classes.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(embeddings, dtype=float)
    y = np.asarray(labels)
    if len(X) < 3 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    sil = float(silhouette_score(X, y))
    kk = min(k, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=kk + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    purity = float(np.mean([(y[nb] == y[i]).mean() for i, nb in enumerate(nbrs)]))
    return sil, purity
