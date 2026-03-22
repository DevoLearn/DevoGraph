#!/usr/bin/env python
"""Proof-of-concept: unified D-GNN pipeline for C. elegans embryogenesis.

Demonstrates that the same CSV can feed both the snapshot-based hypergraph
branch (NDP-HNN) and the continuous-time dynamic graph branch (DevoTG)
through a single ``UnifiedGraphBuilder``, producing compatible node IDs
and features.

Then runs a minimal training loop on the snapshot branch to show the model
actually learns (loss should decrease over epochs).

Usage
-----
    python -m devograph.unified.run_poc                    # from repo root
    python -m devograph.unified.run_poc --epochs 10        # more epochs
    python -m devograph.unified.run_poc --mode both        # build both graphs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from devograph.unified.config import UnifiedConfig
from devograph.unified.graph_builder import UnifiedGraphBuilder
from devograph.unified.model import DevoGNN


def _incidence_bce(node_emb: torch.Tensor, data, device) -> torch.Tensor:
    """BCE reconstruction loss for hyperedge incidence (from NDP-HNN losses.py)."""
    if data.edge_index.numel() == 0:
        return torch.tensor(0.0, device=device)
    row = data.edge_index[0].to(torch.long)
    col = data.edge_index[1].to(torch.long)
    N = data.num_nodes
    E = int(col.max().item()) + 1
    targets = torch.zeros(N, E, device=device)
    targets[row, col] = 1.0
    D = node_emb.size(1)
    he_emb = torch.zeros(E, D, device=device)
    he_emb.index_add_(0, col, node_emb[row])
    cnt = torch.bincount(col, minlength=E).float().clamp_min(1.0).unsqueeze(1)
    he_emb = he_emb / cnt
    logits = node_emb @ he_emb.T
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified D-GNN PoC")
    parser.add_argument("--csv", default=None, help="Path to lineage CSV")
    parser.add_argument("--mode", default="both", choices=["snapshot", "ctdg", "both"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = UnifiedConfig(epochs=args.epochs, lr=args.lr, seed=args.seed)

    # Resolve CSV path relative to repo root
    csv_path = args.csv or cfg.csv_path
    if not Path(csv_path).is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        csv_path = str(repo_root / csv_path)
    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Build both graph representations from one data source ──
    print(f"Loading data from {csv_path}")
    builder = UnifiedGraphBuilder(
        csv_path,
        knn_k=cfg.knn_k,
        spatial_radius=cfg.spatial_radius,
        ctdg_feature_dim=cfg.ctdg_feature_dim,
    )
    print(f"  Cells: {builder.num_cells}  |  T_max: {builder.T_max}")

    graphs = builder.build(mode=args.mode)

    if "snapshots" in graphs:
        snaps = graphs["snapshots"]
        alive_last = int((snaps[-1].t == snaps[-1].t[0]).sum())
        print(
            f"  Snapshots: {len(snaps)} time steps  |  "
            f"Nodes in final snapshot: {snaps[-1].num_nodes}  |  "
            f"Hyperedges at T_max: {snaps[-1].edge_index.size(1)}"
        )

    if "ctdg" in graphs:
        ctdg = graphs["ctdg"]
        print(
            f"  CTDG: {ctdg.num_nodes} nodes  |  "
            f"{ctdg.num_events} division events  |  "
            f"Feature dim: {ctdg.msg.size(1)}"
        )

    # Verify shared node IDs between the two branches
    if "snapshots" in graphs and "ctdg" in graphs:
        assert graphs["snapshots"][0].num_nodes == graphs["ctdg"].num_nodes, (
            "Node count mismatch between snapshot and CTDG branches!"
        )
        print("  Node IDs consistent across both branches.")

    # ── Step 2: Minimal training loop on snapshot branch ───────────────
    if "snapshots" not in graphs:
        print("Skipping training (snapshot mode not selected).")
        return

    snaps = graphs["snapshots"]
    birth_feat = builder.birth_feat
    birth_times = builder.birth_times
    cells = builder.cells

    model = DevoGNN(
        in_dim=cfg.in_dim,
        hid_dim=cfg.hid_dim,
        out_dim=cfg.out_dim,
        num_edge_types=cfg.num_edge_types,
        conv_type=cfg.conv_type,
        rnn_type=cfg.rnn_type,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"\nTraining DevoGNN ({cfg.conv_type}/{cfg.rnn_type}) for {cfg.epochs} epochs on {device}")
    for epoch in range(1, cfg.epochs + 1):
        state = None
        total_loss = 0.0

        for data in snaps:
            data = data.to(device)
            state, pred_xyz, inc_logits = model(data, state)

            t = int(data.t[0].item())
            mask_next = torch.tensor(
                [birth_times.get(c, float("inf")) <= (t + 1) for c in cells],
                dtype=torch.bool,
                device=device,
            )
            target_xyz = torch.tensor(birth_feat[:, :3], device=device)[mask_next]

            loss_xyz = torch.nn.functional.mse_loss(pred_xyz[mask_next], target_xyz)
            h = state[0] if isinstance(state, tuple) else state
            loss_rec = _incidence_bce(h, data, device)
            loss = loss_xyz + loss_rec

            opt.zero_grad()
            loss.backward()
            opt.step()

            if isinstance(state, tuple):
                state = (state[0].detach(), state[1].detach())
            else:
                state = state.detach()
            total_loss += loss.item()

        avg = total_loss / len(snaps)
        print(f"  Epoch {epoch:03d} — avg loss: {avg:.4f}")

    print("\nPoC complete. The unified pipeline works end-to-end.")


if __name__ == "__main__":
    main()
