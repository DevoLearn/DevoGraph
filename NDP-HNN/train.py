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
                device: str = "cuda") -> tuple:

    birth_feat = dataset['birth_feat']
    birth_times = dataset['birth_times']
    cells = dataset['cells']

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {"loss": [], "loss_xyz": [], "loss_rec": []}

    for epoch in range(1, epochs+1):
        state = None
        total_loss = 0.0
        total_xyz  = 0.0
        total_rec  = 0.0

        for data in snapshots:
            data = data.to(device)

            #--- forward one snapshot
            state, pred_xyz, inc_logits = model(data, state)

            #--- mask nodes that are alive at current time step (t)
            t = int(data.t[0].item())
            mask_next = torch.tensor(
                [birth_times[c] <= t for c in cells],
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
            total_xyz  += float(loss_xyz.item())
            total_rec  += float(loss_rec.item())

        n = len(snapshots)
        avg_loss = total_loss / n
        avg_xyz  = total_xyz  / n
        avg_rec  = total_rec  / n

        history["loss"].append(avg_loss)
        history["loss_xyz"].append(avg_xyz)
        history["loss_rec"].append(avg_rec)

        print(f"Epoch {epoch:03d} — loss: {avg_loss:.4f}  xyz: {avg_xyz:.4f}  rec: {avg_rec:.4f}")

    return model, history
