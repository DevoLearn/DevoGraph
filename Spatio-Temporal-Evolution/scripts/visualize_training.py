from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embryo_labels import load_label_map
from src.epic_dataset import EpicEmbryoDataset
from src.estgel_layers import ESTGELClassifier
from scripts.train_estgel import LabeledEmbryoDataset, attach_labels, stratified_split

def plot_curves(history_path: Path, output_path: Path) -> None:
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
        
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Loss Curve
    ax1.plot(epochs, train_loss, color="#1f77b4", marker="o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, color="#ff7f0e", marker="s", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold", pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Accuracy Curve
    ax2.plot(epochs, train_acc, color="#2ca02c", marker="o", label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_acc, color="#d62728", marker="s", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)
    
    fig.suptitle("ESTGEL Classifier Training Performance Curves", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")

def plot_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    # Tick labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel="True Label",
           xlabel="Predicted Label")
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    
    # Annotate entries
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold", fontsize=12)
            
    ax.set_title("ESTGEL Validation Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def evaluate_best_model(checkpoint_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating best model on device: {device}")
    
    # Load config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    processed_dir = Path(config["processed_dir"])
    labels_csv = config["labels_csv"]
    
    label_map = load_label_map(processed_dir, labels_csv=labels_csv)
    dataset = EpicEmbryoDataset(processed_dir, use_global_index=False)
    labels = attach_labels(dataset, label_map)
    
    indices = list(range(len(dataset)))
    if config.get("max_samples") is not None:
        indices = indices[: config["max_samples"]]
        
    subset_labels = [labels[i] for i in indices]
    _, val_idx = stratified_split(subset_labels, config["val_ratio"], config["seed"])
    
    val_subset = LabeledEmbryoDataset(
        dataset,
        [indices[i] for i in val_idx],
        [subset_labels[i] for i in val_idx],
    )
    
    def collate(batch):
        return batch[0]
        
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate)
    
    num_classes = max(subset_labels) + 1
    
    model = ESTGELClassifier(
        num_classes=num_classes,
        K=config["K"],
        max_timesteps=config["max_timesteps"],
        time_stride=config["time_stride"],
        window_size=config["window_size"],
        bptt_truncation=config["bptt_truncation"],
        max_nodes=config["max_nodes"],
        dropout=config["dropout"],
    ).to(device)
    
    # Load weights
    ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    y_true = []
    y_pred = []
    
    for batch in val_loader:
        data = batch.to(device)
        label = int(data.y.item())
        with torch.no_grad():
            logits, _ = model(data)
            pred = int(logits.argmax(dim=-1).item())
        y_true.append(label)
        y_pred.append(pred)
        
    class_names = ["Control (WT)", "Perturbed (RNAi)"]
    print("\n" + "="*60)
    print("ESTGEL Classification Report (Validation Split)")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    print("="*60 + "\n")
    
    # Plot Confusion Matrix
    cm_path = checkpoint_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

def main() -> None:
    checkpoint_dir = REPO_ROOT / "checkpoints" / "estgel"
    
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        print(f"Error: {history_path} does not exist. Did you finish training?")
        sys.exit(1)
        
    curves_path = checkpoint_dir / "training_curves.png"
    plot_curves(history_path, curves_path)
    
    evaluate_best_model(checkpoint_dir)

if __name__ == "__main__":
    main()
