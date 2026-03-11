"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
import torch

def incidence_bce(node_emb, data, device=None):
    """
    Reconstructs the incidence matrix from learned node embeddings.

    Hyperedge embeddings are computed as the mean of their member node
    embeddings (no new parameters). Logits are dot-products between every
    node and every hyperedge embedding, giving an (N, E) score matrix with
    proper gradient flow through node_emb.  Targets are the ground-truth
    binary incidence matrix (constant, no grad).

    Args:
        node_emb: Tensor (N, D) — hidden state returned by the model.
        data:     PyG Data object carrying edge_index and num_nodes.
        device:   optional torch device override.
    """
    dev = device or node_emb.device
    if data.edge_index.numel() == 0:
        return torch.tensor(0.0, device=dev)

    row = data.edge_index[0].to(torch.long)
    col = data.edge_index[1].to(torch.long)
    N   = data.num_nodes
    E   = int(col.max().item()) + 1

    # --- ground-truth binary incidence (constant, no grad) ---
    targets = torch.zeros(N, E, device=dev)
    targets[row, col] = 1.0

    # --- hyperedge embeddings: mean-pool member node embeddings ---
    D = node_emb.size(1)
    he_emb = torch.zeros(E, D, device=dev)
    he_emb.index_add_(0, col, node_emb[row])
    cnt = torch.bincount(col, minlength=E).float().clamp_min(1.0).unsqueeze(1)
    he_emb = he_emb / cnt                          # (E, D) — has grad through node_emb

    # --- (N, E) logits with gradient flow to model parameters ---
    logits = node_emb @ he_emb.T

    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
