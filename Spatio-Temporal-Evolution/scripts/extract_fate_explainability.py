"""
Fate-aware explainability for the trained ESTGEL node classifier.

Runs the fate model over an embryo and, at every recurrence step, reads the EAM
edge attention + the DRL relation strength on each cell-cell edge. Each edge is
annotated with the PREDICTED fate of its endpoints, so we can answer the proposal's
core question: which tissue-tissue interactions the model weights while deciding fate.

Outputs (to checkpoints/estgel_fate/):
  fate_explainability.json  - per-cell fate predictions + tissue-pair interaction stats
  fate_interaction_map.png  - tissue x tissue contact-count and mean-attention heatmaps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cell_fate import FATE_CLASSES, build_fate_index, fate_targets, load_fate_csv
from src.epic_dataset import EpicEmbryoDataset
from src.estgel_node import ESTGELNodeClassifier


def load_model(cfg, device):
    drl = {"Lz": cfg.get("drl_channels") or 12, "Lr": cfg.get("drl_channels") or 12,
           "Lh": cfg.get("drl_channels") or 12} if cfg.get("drl_channels") else {}
    m = ESTGELNodeClassifier(
        num_classes=len(FATE_CLASSES), K=cfg["K"], in_dim=5,
        recurrence_stride=cfg["recurrence_stride"], max_steps=cfg["max_steps"],
        bptt_truncation=cfg["bptt_truncation"], max_nodes=cfg["max_nodes"], **drl,
    ).to(device)
    m.load_state_dict(torch.load(REPO_ROOT / "checkpoints" / "estgel_fate" / "best.pt",
                                 map_location=device)["model_state_dict"])
    m.eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embryo", type=str, default=None, help="npz filename; default = first")
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    ckpt_dir = REPO_ROOT / "checkpoints" / "estgel_fate"
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fi = build_fate_index(); inv = {v: k for k, v in fi.items()}
    cell_fate = load_fate_csv()

    ds = EpicEmbryoDataset(cfg["processed_dir"], use_global_index=False)
    idx = ds.filenames.index(args.embryo) if args.embryo else 0
    data = ds[idx]
    names = [str(c) for c in data.idx_to_cell]
    labels, valid = fate_targets(data.idx_to_cell, cell_fate, fi)

    model = load_model(cfg, device)
    with torch.no_grad():
        logits, collected, timesteps = model(data)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()

    # --- per-cell predictions ---
    per_cell = []
    correct = tot = 0
    for i in range(len(names)):
        rec = {"cell": names[i], "pred_fate": inv[int(pred[i])],
               "confidence": round(float(probs[i, pred[i]]), 4)}
        if valid[i]:
            rec["true_fate"] = inv[int(labels[i])]
            rec["correct"] = bool(pred[i] == labels[i])
            tot += 1; correct += int(pred[i] == labels[i])
        per_cell.append(rec)

    # --- tissue x tissue attention + contact aggregation over all steps ---
    C = len(FATE_CLASSES)
    att_sum = np.zeros((C, C)); cnt = np.zeros((C, C))
    edge_records = []  # (weight, t, src, dst, src_fate, dst_fate)
    for out, t in zip(collected, timesteps):
        ei = out.eam.edge_index.cpu().numpy()
        ew = out.eam.edge_weights.cpu().numpy()
        for e in range(ei.shape[1]):
            s, d = int(ei[0, e]), int(ei[1, e])
            fs, fd = int(pred[s]), int(pred[d])
            w = float(ew[e])
            att_sum[fs, fd] += w; cnt[fs, fd] += 1
            edge_records.append((w, int(t), names[s], names[d], inv[fs], inv[fd]))
    mean_att = np.where(cnt > 0, att_sum / np.maximum(cnt, 1), 0.0)

    # top attended interactions (dedup by cell pair, keep max weight)
    edge_records.sort(key=lambda r: r[0], reverse=True)
    seen = set(); top = []
    for w, t, s, d, fs, fd in edge_records:
        key = (s, d)
        if key in seen:
            continue
        seen.add(key)
        top.append({"src": s, "dst": d, "src_fate": fs, "dst_fate": fd,
                    "attention": round(w, 5), "timestep": t})
        if len(top) >= args.top_k:
            break

    present = [c for c in range(C) if cnt[c].sum() + cnt[:, c].sum() > 0]
    report = {
        "embryo": data.source_file,
        "n_cells": len(names),
        "accuracy_on_labeled": round(correct / max(tot, 1), 4),
        "n_labeled": tot,
        "recurrence_timesteps": timesteps,
        "edge_attention_range": [round(float(min(r[0] for r in edge_records)), 4),
                                  round(float(max(r[0] for r in edge_records)), 4)] if edge_records else None,
        "tissue_pairs_present": [inv[c] for c in present],
        "top_interactions": top,
        "contact_counts": {inv[a]: {inv[b]: int(cnt[a, b]) for b in present if cnt[a, b] > 0}
                            for a in present},
        "mean_attention": {inv[a]: {inv[b]: round(float(mean_att[a, b]), 5)
                                    for b in present if cnt[a, b] > 0} for a in present},
        "per_cell": per_cell,
    }
    out_json = ckpt_dir / "fate_explainability.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- heatmap figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lab = [inv[c] for c in present]
        sub_cnt = cnt[np.ix_(present, present)]
        sub_att = mean_att[np.ix_(present, present)]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=140)
        for ax, mat, title, cmap in [
            (axes[0], np.log1p(sub_cnt), "log(1+contact count)  src -> dst", "viridis"),
            (axes[1], sub_att, "mean EAM attention  src -> dst", "magma"),
        ]:
            im = ax.imshow(mat, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=9)
            ax.set_xlabel("destination fate"); ax.set_ylabel("source fate")
            ax.set_title(title, fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"ESTGEL fate-interaction map — {data.source_file} "
                     f"(acc {report['accuracy_on_labeled']:.2f})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(ckpt_dir / "fate_interaction_map.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)

    print(f"Embryo {data.source_file} | cells {len(names)} | acc {report['accuracy_on_labeled']:.3f}")
    print(f"EAM attention range: {report['edge_attention_range']}")
    print("Tissue pairs present:", report["tissue_pairs_present"])
    print("\nTop attended interactions (src[fate] -> dst[fate]  attn):")
    for t in top[:args.top_k]:
        print(f"  {t['src']:14}[{t['src_fate']:10}] -> {t['dst']:14}[{t['dst_fate']:10}]  {t['attention']:.4f} @t{t['timestep']}")
    print(f"\nSaved: {out_json.name}, fate_interaction_map.png")


if __name__ == "__main__":
    main()
