"""
Train the ESTGEL node-level cell-fate classifier on EPIC embryos.

Targets come from Dataset/fate/cell_fate.csv (see scripts/build_cell_fate.py).
Only cells with a usable fate (not 'mixed'/'unknown') contribute to the loss.
The cell NAME is never a feature — fate is predicted from spatio-temporal signal
(x, y, z, size, blot + graph dynamics) only.

Split is by embryo (held-out embryos), so validation measures whether the learned
spatio-temporal features generalise to unseen embryos. Reports accuracy, macro-F1,
and per-class recall (accuracy alone is misleading under class imbalance).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cell_fate import FATE_CLASSES, build_fate_index, fate_targets, load_fate_csv
from src.epic_dataset import EpicEmbryoDataset
from src.estgel_layers import count_parameters
from src.estgel_node import ESTGELNodeClassifier, compute_feature_stats


def build_targets(dataset: EpicEmbryoDataset, cell_fate, fate_index):
    """Precompute (labels, valid) per embryo, aligned to each embryo's node order."""
    out = []
    for i in range(len(dataset)):
        idx_to_cell = np.load(
            dataset.processed_dir / dataset.filenames[i], allow_pickle=True
        )["idx_to_cell"]
        labels, valid = fate_targets(idx_to_cell, cell_fate, fate_index)
        out.append((labels, valid))
    return out


def global_class_weights(targets, num_classes, present_ids, cap: float = 5.0):
    """Mild sqrt-inverse-frequency weights, normalised to mean 1 and capped.

    Raw inverse frequency gives ultra-rare classes (e.g. coelomocyte ~0.2%) 100x+
    the weight of common ones, which makes the model predict rare classes
    everywhere and collapses accuracy. The sqrt softens this; the cap bounds it.
    """
    counts = torch.zeros(num_classes)
    for labels, valid in targets:
        for c in labels[valid]:
            counts[c] += 1
    w = torch.zeros(num_classes)
    inv = torch.zeros(len(present_ids))
    for i, c in enumerate(present_ids):
        inv[i] = (1.0 / counts[c].clamp(min=1.0)).sqrt()
    inv = inv / inv.mean()          # mean-1 normalise
    inv = inv.clamp(max=cap)        # bound the largest weight
    for i, c in enumerate(present_ids):
        w[c] = inv[i]
    return w


@torch.no_grad()
def evaluate(model, dataset, val_idx, targets, device, num_classes):
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    loss_sum, n = 0.0, 0
    for i in val_idx:
        data = dataset[i]
        labels, valid = targets[i]
        if not valid.any():
            continue
        logits, _, _ = model(data)
        y = torch.tensor(labels, device=device)
        vmask = torch.tensor(valid, device=device)
        loss_sum += F.cross_entropy(logits[vmask], y[vmask]).item()
        n += 1
        pred = logits[vmask].argmax(1).cpu()
        yv = y[vmask].cpu()
        for t, p in zip(yv.tolist(), pred.tolist()):
            conf[t, p] += 1
    correct = conf.diag().sum().item()
    total = conf.sum().item()
    acc = correct / max(total, 1)
    # macro-F1 over classes that appear as a true label
    f1s = []
    for c in range(num_classes):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"loss": loss_sum / max(n, 1), "acc": acc, "macro_f1": macro_f1, "n": n}, conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Dataset" / "processed" / "by_embryo")
    ap.add_argument("--fate-csv", type=Path, default=REPO_ROOT / "Dataset" / "fate" / "cell_fate.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None)
    # ESTGEL recurrence is strided across the FULL developmental time so every cell
    # (incl. late-born) is captured; each cell is pooled over its own alive steps.
    ap.add_argument("--recurrence-stride", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--bptt-truncation", type=int, default=6)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--max-nodes", type=int, default=2500)  # max embryo has 2434 cells
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Skip embryos with more than this many cells (VRAM guard; DRL is O(channels*N^2)).")
    ap.add_argument("--drl-channels", type=int, default=None,
                    help="Set Lz=Lr=Lh (default 12). Lower => less VRAM/time.")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--resume", action="store_true", help="warm-start weights from best.pt and continue the run/history")
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay over this run's epochs")
    ap.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints" / "estgel_fate")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    fate_index = build_fate_index()
    num_classes = len(FATE_CLASSES)
    cell_fate = load_fate_csv(args.fate_csv)

    dataset = EpicEmbryoDataset(args.processed_dir, use_global_index=False)
    targets = build_targets(dataset, cell_fate, fate_index)

    indices = list(range(len(dataset)))
    if args.max_cells is not None:
        kept = [i for i in indices if len(targets[i][0]) <= args.max_cells]
        skipped = len(indices) - len(kept)
        if skipped:
            print(f"[max-cells={args.max_cells}] skipping {skipped} embryos over the cell cap.")
        indices = kept
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_ratio))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    present_ids = sorted({int(c) for i in indices for c in targets[i][0][targets[i][1]]})
    inv = {v: k for k, v in fate_index.items()}
    weights = None if args.no_class_weight else global_class_weights(targets, num_classes, present_ids).to(device)

    drl_kwargs = {}
    if args.drl_channels is not None:
        drl_kwargs = {"Lz": args.drl_channels, "Lr": args.drl_channels, "Lh": args.drl_channels}
    model = ESTGELNodeClassifier(
        num_classes=num_classes, K=args.K, in_dim=5,
        recurrence_stride=args.recurrence_stride, max_steps=args.max_steps,
        bptt_truncation=args.bptt_truncation,
        max_nodes=args.max_nodes, dropout=args.dropout, **drl_kwargs,
    ).to(device)

    # Per-feature standardization from the TRAIN split only (saved in checkpoint buffers).
    fmean, fstd = compute_feature_stats(dataset, train_idx)
    model.set_feature_stats(fmean, fstd)
    print(f"Feature stats (train): mean={[round(v,2) for v in fmean.tolist()]} "
          f"std={[round(v,2) for v in fstd.tolist()]}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None

    start_epoch, best_f1, history = 1, -1.0, []
    if args.resume:
        ck = args.checkpoint_dir / "best.pt"
        if ck.exists():
            prev = torch.load(ck, map_location=device)
            model.load_state_dict(prev["model_state_dict"])
            best_f1 = float(prev.get("val_macro_f1", -1.0))
            hp = args.checkpoint_dir / "history.json"
            if hp.exists():
                history = json.loads(hp.read_text())
                start_epoch = len(history) + 1
            print(f"Resumed from {ck}: prev best macroF1={best_f1:.3f}, continuing at epoch {start_epoch}")
        else:
            print(f"--resume given but {ck} not found; training fresh.")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 72)
    print("ESTGEL Cell-Fate (node) Training")
    print("=" * 72)
    print(f"Device: {device} | embryos: {len(indices)} (train={len(train_idx)}, val={len(val_idx)})")
    print(f"Classes present: {[inv[c] for c in present_ids]}")
    print(f"Params: {count_parameters(model):,} | recurrence: stride {args.recurrence_stride}, <= {args.max_steps} steps (full-T, lifetime-pooled)")
    print()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum, n = 0.0, 0
        for i in tqdm(train_idx, desc=f"epoch {epoch}", leave=False):
            data = dataset[i]
            labels, valid = targets[i]
            if not valid.any():
                continue
            logits, _, _ = model(data)
            y = torch.tensor(labels, device=device)
            vmask = torch.tensor(valid, device=device)
            loss = F.cross_entropy(logits[vmask], y[vmask], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
            n += 1
        val, conf = evaluate(model, dataset, val_idx, targets, device, num_classes)
        elapsed = time.perf_counter() - t0
        row = {"epoch": epoch, "train_loss": loss_sum / max(n, 1),
               "val_loss": val["loss"], "val_acc": val["acc"],
               "val_macro_f1": val["macro_f1"], "elapsed_s": elapsed}
        history.append(row)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train loss {row['train_loss']:.3f} | "
              f"val loss {val['loss']:.3f} acc {val['acc']:.3f} macroF1 {val['macro_f1']:.3f} | "
              f"lr {lr_now:.1e} | {elapsed:.1f}s")
        if scheduler:
            scheduler.step()

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config,
                "val_macro_f1": val["macro_f1"]}
        torch.save(ckpt, args.checkpoint_dir / "last.pt")
        if val["macro_f1"] >= best_f1:
            best_f1 = val["macro_f1"]
            torch.save(ckpt, args.checkpoint_dir / "best.pt")
            # per-class recall snapshot for the best model
            rec = {inv[c]: round((conf[c, c] / conf[c].sum()).item(), 3)
                   for c in range(num_classes) if conf[c].sum() > 0}
            (args.checkpoint_dir / "per_class_recall.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Done. Best val macro-F1: {best_f1:.3f} | checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
