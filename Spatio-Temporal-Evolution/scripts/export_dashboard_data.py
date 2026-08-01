"""
Export a single self-contained data blob for the fate dashboard.

Runs the trained ESTGEL fate model on one embryo and packages:
  - per-cell predicted fate + confidence + true fate + 3D positions over time
  - developmental frames (timesteps)
  - global explainability (feature occlusion, graph ablation, tissue interaction map)
  - training history + per-class recall

Writes dashboard/fate_dashboard_data.json
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
from src.epic_preprocess import lineage_parent
from src.estgel_node import ESTGELNodeClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embryo", type=str, default="CD011505_end1red_bright.npz")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--max-edges", type=int, default=650, help="cap contact edges drawn per frame")
    args = ap.parse_args()

    ckpt = REPO_ROOT / "checkpoints" / "estgel_fate"
    cfg = json.loads((ckpt / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fi = build_fate_index(); inv = {v: k for k, v in fi.items()}
    cell_fate = load_fate_csv()

    ds = EpicEmbryoDataset(cfg["processed_dir"], use_global_index=False)
    idx = ds.filenames.index(args.embryo) if args.embryo in ds.filenames else 0
    data = ds[idx]
    names = [str(c) for c in data.idx_to_cell]
    labels, valid = fate_targets(data.idx_to_cell, cell_fate, fi)

    # Raw (un-normalized) features straight from the npz — model() mutates data.x in place.
    raw = np.load(ds.processed_dir / ds.filenames[idx], allow_pickle=True)
    X = raw["X"]                  # (N, d, T)  raw x,y,z,size,blot
    am = raw["alive_mask"]        # (N, T)

    drl = {"Lz": cfg["drl_channels"], "Lr": cfg["drl_channels"], "Lh": cfg["drl_channels"]} if cfg.get("drl_channels") else {}
    model = ESTGELNodeClassifier(
        num_classes=len(FATE_CLASSES), K=cfg["K"], in_dim=5,
        recurrence_stride=cfg["recurrence_stride"], max_steps=cfg["max_steps"],
        bptt_truncation=cfg["bptt_truncation"], max_nodes=cfg["max_nodes"], **drl,
    ).to(device)
    model.load_state_dict(torch.load(ckpt / "best.pt", map_location=device)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(1)

    # choose evenly spaced frames where cells are alive
    T = X.shape[2]
    alive_t = [t for t in range(T) if am[:, t].any()]
    if len(alive_t) > args.n_frames:
        sel = np.linspace(0, len(alive_t) - 1, args.n_frames).round().astype(int)
        frames = [alive_t[i] for i in sorted(set(sel.tolist()))]
    else:
        frames = alive_t

    name_to_idx = {n: i for i, n in enumerate(names)}
    cells = []
    for i in range(len(names)):
        alive_frames = [fi for fi, t in enumerate(frames) if am[i, t]]
        pos = [[round(float(X[i, 0, t]), 1), round(float(X[i, 1, t]), 1), round(float(X[i, 2, t]), 1)]
               if am[i, t] else None for t in frames]
        sizes_alive = [float(X[i, 3, t]) for t in frames if am[i, t]]
        parent = lineage_parent(names[i])
        rec = {
            "name": names[i],
            "fate": inv[int(pred[i])],
            "conf": round(float(probs[i, pred[i]]), 3),
            "size": round(sum(sizes_alive) / len(sizes_alive), 1) if sizes_alive else 0.0,
            "b": alive_frames[0] if alive_frames else 0,          # birth frame index
            "p": name_to_idx.get(parent, -1) if parent else -1,   # parent cell index
            "pos": pos,
        }
        if valid[i]:
            rec["true_fate"] = inv[int(labels[i])]
            rec["correct"] = bool(pred[i] == labels[i])
        cells.append(rec)

    # per-frame contact edges (undirected, capped) straight from the model's graph
    e_src, e_dst, e_t = raw["edge_src"], raw["edge_dst"], raw["edge_t"]
    edges_per_frame = []
    for t in frames:
        m = e_t == t
        pairs = set()
        for s, d in zip(e_src[m].tolist(), e_dst[m].tolist()):
            # keep only co-alive spatial contacts (both endpoints present this frame);
            # lineage edges to an already-divided parent aren't drawable.
            if s != d and am[s, t] and am[d, t]:
                pairs.add((s, d) if s < d else (d, s))
                if len(pairs) >= args.max_edges:
                    break
        edges_per_frame.append([[a, b] for a, b in pairs])

    def load_json(p):
        p = ckpt / p
        return json.loads(p.read_text()) if p.exists() else None

    blob = {
        "embryo": data.source_file,
        "n_cells": len(names),
        "frames": frames,
        "fate_classes": list(FATE_CLASSES),
        "cells": cells,
        "edges": edges_per_frame,
        "metrics": {
            "history": load_json("history.json"),
            "per_class_recall": load_json("per_class_recall.json"),
        },
        "explainability": {
            "saliency": load_json("fate_saliency.json"),
            "interactions": load_json("fate_explainability.json"),
        },
    }
    blob_json = json.dumps(blob)
    out = REPO_ROOT / "dashboard" / "fate_dashboard_data.json"
    out.write_text(blob_json, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} KB) | embryo {data.source_file} | "
          f"{len(names)} cells | {len(frames)} frames {frames}")

    # Inject into the template -> self-contained dashboard/index.html
    template = REPO_ROOT / "dashboard" / "fate_dashboard.template.html"
    if template.exists():
        html = template.read_text(encoding="utf-8").replace("__FATE_DATA__", blob_json)
        (REPO_ROOT / "dashboard" / "index.html").write_text(html, encoding="utf-8")
        print(f"Built dashboard/index.html ({len((REPO_ROOT/'dashboard'/'index.html').read_text(encoding='utf-8'))/1024:.0f} KB, self-contained)")
    else:
        print(f"Template not found ({template}); skipped index.html build.")


if __name__ == "__main__":
    main()
