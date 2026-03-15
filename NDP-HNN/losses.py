"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
import torch

def incidence_bce(pred_logits, data, device=None):
    """
    BCE reconstruction loss for the hyperedge incidence matrix.

    pred_logits: (N, E) float tensor produced by the model decoder.
                 Must be connected to the computation graph so gradients flow.
    data:        PyG Data object with edge_index[0]=node ids, edge_index[1]=hyperedge ids.

    Builds a binary target (N, E) from edge_index (1 for known memberships,
    0 elsewhere) and computes BCE-with-logits against pred_logits.
    Skips if there are no hyperedges.
    """
    if data.edge_index.numel() == 0 or pred_logits.numel() == 0:
        return torch.tensor(0.0, device=device or data.x.device)
    col = data.edge_index[1] - int(data.edge_index[1].min().item())
    target = torch.zeros_like(pred_logits)
    target[data.edge_index[0], col] = 1.0
    return torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target)
