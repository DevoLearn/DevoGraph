"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
from typing import List
import numpy as np
import torch
import os

def extract_embeddings(model, snapshots: List, device: str = "cuda") -> np.ndarray:
    """
    Runs the model over time and returns a (T, N, D) numpy array
    of hidden states. (D == model hidden size)
    """
    model.eval()
    state = None
    embeds = []

    with torch.no_grad():
        for data in snapshots:
            data = data.to(device)
            state, _, _ = model(data, state)

            #--- take H from state
            if isinstance(state, tuple):  # LSTM (h, c)
                h = state[0]
            else:
                h = state
            h_cpu = h.detach().cpu().numpy()
            embeds.append(h_cpu)
            #--- keep only the next state's device tensor to continue
            if isinstance(state, tuple):
                state = (state[0].detach(), state[1].detach())
            else:
                state = state.detach()
            torch.cuda.empty_cache()

    return np.stack(embeds, axis=0)  # (T, N, D)

def save_embeddings(embeds: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeds)
    print(f">>> Saved embeddings to: {path} shape={embeds.shape}")
