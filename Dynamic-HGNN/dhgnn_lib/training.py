"""Loss, metrics, and per-timepoint train/eval step for the DHGNN.

The model is trained on a single supervised objective, **cell fate**
(founder-lineage classification):

    L_fate   (cross-entropy)

"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FateLoss(nn.Module):
    """Cross-entropy over founder-lineage classes, on present + fate-train cells."""

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        fate: torch.Tensor,
        present_mask: torch.Tensor,
        fate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        present = present_mask.bool()
        fmask = present if fate_mask is None else (present & fate_mask.bool())
        if fmask.sum() == 0:
            zero = outputs["fate_logits"].sum() * 0.0
            return {"fate": zero}
        fate_loss = F.cross_entropy(outputs["fate_logits"][fmask], fate[fmask])
        return {"fate": fate_loss}


@torch.no_grad()
def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    fate: torch.Tensor,
    present_mask: torch.Tensor,
    fate_eval_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Fate accuracy on the (held-out) fate-eval cells that are present."""
    present = present_mask.bool()
    emask = present if fate_eval_mask is None else (present & fate_eval_mask.bool())
    if emask.sum() == 0:
        return {"fate_acc": 0.0, "n_eval": 0}
    pred = outputs["fate_logits"][emask].argmax(dim=-1)
    return {"fate_acc": (pred == fate[emask]).float().mean().item(), "n_eval": int(emask.sum().item())}


def build_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, t_max: int, eta_min: float = 1e-5):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def run_timepoint(
    model: nn.Module,
    x: torch.Tensor,
    batch,
    fate: torch.Tensor,
    present_mask: torch.Tensor,
    loss_fn: FateLoss,
    optimizer: Optional[torch.optim.Optimizer] = None,
    universe=None,
    static_sets=None,
    recluster: bool = False,
    fate_train_mask: Optional[torch.Tensor] = None,
    fate_eval_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Run one timepoint forward (+ backward if ``optimizer`` is given).

    ``fate_train_mask`` / ``fate_eval_mask`` implement the cell-disjoint fate
    split (supervise on one set of cells, evaluate on the held-out set).
    """
    train = optimizer is not None
    model.train(train)

    with torch.set_grad_enabled(train):
        outputs = model(
            x, batch, universe=universe, present_mask=present_mask, static_sets=static_sets, recluster=recluster
        )
        losses = loss_fn(outputs, fate, present_mask, fate_mask=fate_train_mask)

        if train:
            optimizer.zero_grad()
            losses["fate"].backward()
            optimizer.step()

    metrics = compute_metrics(outputs, fate, present_mask, fate_eval_mask=fate_eval_mask)
    log = {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
    log.update(metrics)
    return log
