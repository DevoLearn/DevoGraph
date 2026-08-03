"""
ESTGEL node-level cell-fate classifier.

Reuses the ESTGEL EAM -> DRL -> DNL timestep block, but instead of pooling to a
single graph label it keeps the per-node refined states X_hat and predicts a fate
class for every cell. This matches the reframed target: predict each cell's terminal
fate from its spatio-temporal dynamics (position, timing, contacts, expression).

Node representation (v2 — lifetime pooling):
  The ESTGEL recurrence is run STRIDED across the *full* developmental time
  (t = 0, s, 2s, ... up to T), carrying DRL/DNL state. For each cell i we pool its
  refined state X_hat[i] over exactly the strided steps where i is alive:

      emb[i] = concat( mean_t alive X_hat[i, t] ,  X_hat[i, last-alive t] )

  This gives every cell a real, informative embedding regardless of when it is born
  — crucial because ~70% of EPIC cells are born late; a fixed early window leaves
  them all-zero and forces an all-majority ("neuron") predictor.

IMPORTANT (training): fate is a deterministic function of the *cell name*. Do NOT
feed the name as a model feature — that is trivial leakage. Predict from position /
timing / contacts / expression only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.estgel_layers import DNLOutput, ESTGELTimestepBlock


def compute_feature_stats(dataset, indices, in_dim: int = 5):
    """Per-feature mean/std over alive cells across the given embryos (train split only)."""
    sums = torch.zeros(in_dim)
    sqs = torch.zeros(in_dim)
    cnt = 0
    for i in indices:
        z = np.load(dataset.processed_dir / dataset.filenames[i], allow_pickle=True)
        X = torch.tensor(z["X"], dtype=torch.float32)          # (N, d, T)
        am = torch.tensor(z["alive_mask"], dtype=torch.bool)    # (N, T)
        for di in range(in_dim):
            v = X[:, di, :][am]
            sums[di] += v.sum()
            sqs[di] += (v * v).sum()
        cnt += int(am.sum())
    mean = sums / max(cnt, 1)
    var = (sqs / max(cnt, 1) - mean ** 2).clamp(min=1e-12)
    return mean, var.sqrt()


class ESTGELNodeClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        K: int = 11,
        in_dim: int = 5,
        recurrence_stride: int = 8,
        max_steps: int = 32,
        bptt_truncation: int = 6,
        head_hidden: int = 128,
        dropout: float = 0.5,
        max_nodes: int = 3072,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.recurrence_stride = recurrence_stride
        self.max_steps = max_steps
        self.bptt_truncation = bptt_truncation
        self.max_nodes = max_nodes

        self.block = ESTGELTimestepBlock(K=K, in_dim=in_dim, max_nodes=max_nodes, **block_kwargs)

        # Per-feature standardization. EPIC stores raw physical values (x,y ~ hundreds,
        # blot up to ~1e5); without this the linear head explodes and training diverges.
        # Set from training data via set_feature_stats(); saved in the checkpoint so
        # inference uses identical stats.
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

        # embedding = ESTGEL X_hat (mean+last) + RAW features (mean+last) + birth/lifespan.
        # Raw 3D position/expression is the strongest fate signal and legitimate input
        # (only the cell NAME is off-limits); X_hat carries the learned relational state.
        node_dim = in_dim * 4 + 2
        self.head = nn.Sequential(
            nn.Linear(node_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install per-feature standardization stats (shape (in_dim,))."""
        self.feat_mean.copy_(mean.to(self.feat_mean).view(-1))
        self.feat_std.copy_(std.to(self.feat_std).view(-1).clamp(min=1e-6))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize (N, d, T) features; padding is re-zeroed downstream by alive masks."""
        return (x - self.feat_mean.view(1, -1, 1)) / self.feat_std.view(1, -1, 1)

    def _recurrence_timesteps(self, data: Data) -> list[int]:
        """Strided timesteps across the full alive span (each must have live cells)."""
        T = int(data.T)
        alive_any = data.alive_mask.any(dim=0)
        steps = [t for t in range(0, T, max(self.recurrence_stride, 1)) if bool(alive_any[t])]
        if not steps:
            return [0]
        if len(steps) > self.max_steps:  # subsample evenly to bound compute
            step = (len(steps) - 1) / (self.max_steps - 1)
            steps = sorted({steps[int(round(i * step))] for i in range(self.max_steps)})
        return steps

    def _pool_lifetime(
        self,
        collected: list[DNLOutput],
        timesteps: list[int],
        data: Data,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-node embedding: ESTGEL X_hat (mean+last) ++ raw features (mean+last) ++ birth/lifespan."""
        N, d = data.x.shape[0], self.in_dim
        node_dim = d * 4 + 2
        if not collected:
            return torch.zeros(N, node_dim, device=device)

        # --- ESTGEL refined state, pooled over the strided steps where the cell is alive ---
        Xs = torch.stack([o.X_hat for o in collected], dim=0)   # (S, N, d)
        Al = torch.stack([data.alive_mask[:, t].to(device).float() for t in timesteps], dim=0)  # (S,N)
        cnt = Al.sum(dim=0).clamp(min=1.0)
        xhat_mean = (Xs * Al.unsqueeze(-1)).sum(dim=0) / cnt.unsqueeze(-1)
        S = Xs.shape[0]
        last_step = (Al * torch.arange(1, S + 1, device=device).float().unsqueeze(1)).argmax(dim=0)
        xhat_last = Xs[last_step, torch.arange(N, device=device)]

        # --- Raw features, pooled over the FULL alive span (all T; no recurrence needed) ---
        Xraw = data.x.to(device)                                # (N, d, T)
        am = data.alive_mask.to(device).float()                 # (N, T)
        rcnt = am.sum(dim=1).clamp(min=1.0)                     # (N,)
        raw_mean = (Xraw * am.unsqueeze(1)).sum(dim=2) / rcnt.unsqueeze(1)   # (N, d)
        Tlen = am.shape[1]
        last_t = (am * torch.arange(1, Tlen + 1, device=device).float()).argmax(dim=1)  # (N,)
        raw_last = Xraw[torch.arange(N, device=device), :, last_t]           # (N, d)
        birth = (am.argmax(dim=1).float() / max(Tlen, 1)).unsqueeze(1)       # (N,1)
        lifespan = (rcnt / max(Tlen, 1)).unsqueeze(1)                        # (N,1)

        return torch.cat([xhat_mean, xhat_last, raw_mean, raw_last, birth, lifespan], dim=1)

    def forward(
        self,
        data: Data,
        *,
        mode: str = "sparse",
    ) -> tuple[torch.Tensor, list[DNLOutput], list[int]]:
        device = next(self.parameters()).device
        data = data.to(device)
        data.x = self._normalize(data.x)  # both the ESTGEL block and pooling read data.x

        timesteps = self._recurrence_timesteps(data)
        collected: list[DNLOutput] = []
        A_hat_prev: torch.Tensor | None = None
        X_hat_prev: torch.Tensor | None = None

        steps_since_detach = 0
        for t_idx in timesteps:
            out = self.block.forward_timestep(
                data, t_idx, A_hat_prev=A_hat_prev, X_hat_prev=X_hat_prev,
                mode=mode, device=device,
            )
            collected.append(out)
            steps_since_detach += 1
            detach = (not self.training) or (steps_since_detach >= self.bptt_truncation)
            if detach:
                A_hat_prev = out.A_hat.detach()
                X_hat_prev = out.X_hat.detach()
                steps_since_detach = 0
            else:
                A_hat_prev = out.A_hat
                X_hat_prev = out.X_hat

        node_emb = self._pool_lifetime(collected, timesteps, data, device)
        node_logits = self.head(node_emb)  # (N, num_classes)
        return node_logits, collected, timesteps
