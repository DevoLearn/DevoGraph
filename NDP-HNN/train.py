"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
from typing import List, Dict, Any
import torch
from losses import incidence_bce

def train_model(model,
                snapshots: List,
                dataset: Dict[str, Any],
                epochs: int = 30,
                lr: float = 1e-3,
                device: str = "cuda"):

    birth_feat = dataset['birth_feat']
    birth_times = dataset['birth_times']
    cells = dataset['cells']

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        state = None
        total_loss = 0.0

        for data in snapshots:
            data = data.to(device)

            #--- forward one snapshot
            state, pred_xyz, inc_logits = model(data, state)

            #--- mask nodes that are alive at next time step (t+1)
            t = int(data.t[0].item())
            mask_next = torch.tensor(
                [birth_times[c] <= (t + 1) for c in cells],
                dtype=torch.bool, device=device
            )
            target_xyz = torch.tensor(birth_feat[:, :3], device=device)[mask_next]

            #--- combine your two losses
            loss_xyz = torch.nn.functional.mse_loss(pred_xyz[mask_next], target_xyz)
            # Extract node embeddings (h) from state; for LSTM state is (h, c)
            h = state[0] if isinstance(state, tuple) else state
            loss_rec = incidence_bce(h, data, device=device)
            loss = loss_xyz + loss_rec

            opt.zero_grad()
            loss.backward()
            opt.step()

            #--- detach hidden state to truncate graph
            if isinstance(state, tuple):  # [LSTM]
                state = (state[0].detach(), state[1].detach())
            else:
                state = state.detach()
            total_loss += float(loss.item())

        print(f"Epoch {epoch:03d} — avg loss: {total_loss/len(snapshots):.4f}")

    return model
