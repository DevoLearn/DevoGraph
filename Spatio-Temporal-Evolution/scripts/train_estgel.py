from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embryo_labels import load_label_map
from src.epic_dataset import EpicEmbryoDataset
from src.estgel_layers import ESTGELClassifier, count_parameters


class LabeledEmbryoDataset(torch.utils.data.Dataset):
    """Wraps EpicEmbryoDataset with per-sample classification labels."""

    def __init__(self, base: EpicEmbryoDataset, indices: list[int], labels: list[int]) -> None:
        self.base = base
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        data = self.base[self.indices[idx]]
        data.y = torch.tensor([self.labels[idx]], dtype=torch.long)
        return data


def stratified_split(
    labels: list[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label, indices in by_class.items():
        rng.shuffle(indices)
        if len(indices) == 1:
            train_idx.extend(indices)
            continue
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return train_idx, val_idx


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for y in labels:
        counts[y] += 1
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


@torch.no_grad()
def evaluate(
    model: ESTGELClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in loader:
        data = batch.to(device)
        label = data.y.view(-1).to(device)
        logits, _ = model(data)
        loss = F.cross_entropy(logits.unsqueeze(0), label)
        loss_sum += loss.item()
        pred = logits.argmax(dim=-1)
        correct += int((pred == label).sum().item())
        total += 1

    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n": float(total),
    }


def train_one_epoch(
    model: ESTGELClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weight: torch.Tensor | None,
) -> dict[str, float]:
    model.train()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        data = batch.to(device)
        label = data.y.view(-1).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data)
        loss = F.cross_entropy(
            logits.unsqueeze(0),
            label,
            weight=class_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item()
        pred = logits.argmax(dim=-1)
        correct += int((pred == label).sum().item())
        total += 1

    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n": float(total),
    }


def attach_labels(dataset: EpicEmbryoDataset, label_map: dict[str, int]) -> list[int]:
    labels: list[int] = []
    for name in dataset.filenames:
        if name not in label_map:
            raise KeyError(f"No label for {name}")
        labels.append(label_map[name])
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ESTGEL embryo classifier")
    ap.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "Dataset" / "processed" / "by_embryo")
    ap.add_argument("--labels-csv", type=Path, default=None, help="Optional filename,label CSV override")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick runs")
    ap.add_argument("--max-timesteps", type=int, default=6)
    ap.add_argument("--time-stride", type=int, default=25)
    ap.add_argument("--window-size", type=int, default=20)
    ap.add_argument("--bptt-truncation", type=int, default=8)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--max-nodes", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints" / "estgel")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    label_map = load_label_map(args.processed_dir, labels_csv=args.labels_csv)
    dataset = EpicEmbryoDataset(args.processed_dir, use_global_index=False)
    labels = attach_labels(dataset, label_map)

    indices = list(range(len(dataset)))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    subset_labels = [labels[i] for i in indices]
    train_idx, val_idx = stratified_split(subset_labels, args.val_ratio, args.seed)
    train_subset = LabeledEmbryoDataset(
        dataset,
        [indices[i] for i in train_idx],
        [subset_labels[i] for i in train_idx],
    )
    val_subset = LabeledEmbryoDataset(
        dataset,
        [indices[i] for i in val_idx],
        [subset_labels[i] for i in val_idx],
    )

    def collate(batch):
        return batch[0]

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate)

    num_classes = max(subset_labels) + 1
    weights = class_weights(subset_labels, num_classes).to(device)

    model = ESTGELClassifier(
        num_classes=num_classes,
        K=args.K,
        max_timesteps=args.max_timesteps,
        time_stride=args.time_stride,
        window_size=args.window_size,
        bptt_truncation=args.bptt_truncation,
        max_nodes=args.max_nodes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["processed_dir"] = str(config["processed_dir"])
    config["checkpoint_dir"] = str(config["checkpoint_dir"])
    if config["labels_csv"] is not None:
        config["labels_csv"] = str(config["labels_csv"])
    (args.checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 72)
    print("ESTGEL Training")
    print("=" * 72)
    print(f"Device:          {device}")
    print(f"Samples:         {len(indices)} (train={len(train_subset)}, val={len(val_subset)})")
    print(f"Label balance:   {dict((l, subset_labels.count(l)) for l in sorted(set(subset_labels)))}")
    print(f"Class weights:   {weights.tolist()}")
    print(f"Parameters:      {count_parameters(model):,}")
    print(f"Max timesteps:   {args.max_timesteps} (stride={args.time_stride})")
    print(f"Window / BPTT:   {args.window_size} / {args.bptt_truncation}")
    print(f"Max nodes:       {args.max_nodes}")
    print()

    best_val_acc = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, weights)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "elapsed_s": elapsed,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} | "
            f"{elapsed:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_metrics["accuracy"],
            "config": config,
        }
        torch.save(ckpt, args.checkpoint_dir / "last.pt")
        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(ckpt, args.checkpoint_dir / "best.pt")

    (args.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Training complete. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
