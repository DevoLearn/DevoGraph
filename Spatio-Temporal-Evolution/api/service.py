"""
ESTGEL cell-fate prediction API.

Accepts an EPIC-format CSV of a C. elegans embryo, runs the trained node
classifier, and returns predictions plus analysis in exactly the JSON shape the
dashboard already consumes, so the front end can swap the payload and redraw.

POST /predict   multipart file=<csv>   [?ablation=true]
GET  /health
GET  /schema    the expected CSV columns

Deployed as a Hugging Face Space (Docker SDK); see README.md.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cell_fate import (  # noqa: E402
    FATE_CLASSES, build_fate_index, fate_targets, load_fate_csv,
)
from src.epic_preprocess import lineage_parent, preprocess_epic_file_sparse  # noqa: E402
from src.estgel_node import ESTGELNodeClassifier  # noqa: E402

REQUIRED_COLUMNS = ["cell", "time", "x", "y", "z", "size", "blot"]

# Guards: this runs on a small shared CPU, and cost grows with N^2 per timestep.
MAX_CELLS = int(os.getenv("MAX_CELLS", "1200"))
# The EAM graph decomposition splits the graph into K nested subgraphs, so it needs
# at least K nodes. K is fixed at 11 by the trained checkpoint and must not be
# clamped at inference, since that would change the decomposition the model expects.
MIN_CELLS = 12
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
N_FRAMES = int(os.getenv("N_FRAMES", "30"))
MAX_EDGES_PER_FRAME = 650

app = FastAPI(title="ESTGEL cell-fate API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://barshan.is-a.dev",
        "https://blueee04.github.io",
        "http://localhost:1313",   # hugo dev server
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_MODEL: ESTGELNodeClassifier | None = None
_FATE_MAP: dict[str, str] | None = None
_STATIC_METRICS: dict | None = None
_CACHE: dict[str, dict] = {}


def _load():
    """Lazy-load the checkpoint once per process."""
    global _MODEL, _FATE_MAP, _STATIC_METRICS
    if _MODEL is not None:
        return
    import json

    ckpt_path = ROOT / "model" / "best.pt"
    cfg = json.loads((ROOT / "model" / "config.json").read_text())
    drl = {}
    if cfg.get("drl_channels"):
        c = cfg["drl_channels"]
        drl = {"Lz": c, "Lr": c, "Lh": c}
    model = ESTGELNodeClassifier(
        num_classes=len(FATE_CLASSES), K=cfg["K"], in_dim=5,
        recurrence_stride=cfg["recurrence_stride"], max_steps=cfg["max_steps"],
        bptt_truncation=cfg["bptt_truncation"], max_nodes=cfg["max_nodes"], **drl,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    torch.set_grad_enabled(False)

    _MODEL = model
    _FATE_MAP = load_fate_csv(ROOT / "model" / "cell_fate.csv")
    hist_p, rec_p = ROOT / "model" / "history.json", ROOT / "model" / "per_class_recall.json"
    _STATIC_METRICS = {
        "history": json.loads(hist_p.read_text()) if hist_p.exists() else None,
        "per_class_recall": json.loads(rec_p.read_text()) if rec_p.exists() else None,
    }


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            400,
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected EPIC format with: {', '.join(REQUIRED_COLUMNS)}.",
        )
    if df.empty:
        raise HTTPException(400, "CSV has no rows.")
    n_cells = df["cell"].astype(str).nunique()
    if n_cells < MIN_CELLS:
        raise HTTPException(
            400,
            f"Embryo has only {n_cells} distinct cells. The edge-attention module "
            f"decomposes the graph into 11 nested subgraphs, so at least {MIN_CELLS} "
            f"cells are needed.",
        )
    if n_cells > MAX_CELLS:
        raise HTTPException(
            413,
            f"Embryo has {n_cells} cells, above the {MAX_CELLS} limit for this "
            f"hosted demo. Run the model locally for larger embryos.",
        )
    for col in ["time", "x", "y", "z", "size", "blot"]:
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")):
            raise HTTPException(400, f"Column '{col}' must be numeric.")


def _build_graph(X, alive_mask, e_src, e_dst, e_t) -> Data:
    edge_index = (
        torch.stack([torch.tensor(e_src, dtype=torch.long), torch.tensor(e_dst, dtype=torch.long)])
        if len(e_src) else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        alive_mask=torch.tensor(alive_mask, dtype=torch.bool),
        edge_index=edge_index,
        edge_t=torch.tensor(e_t, dtype=torch.long),
        T=X.shape[2],
    )


def _pick_frames(alive_mask, n_frames: int) -> list[int]:
    T = alive_mask.shape[1]
    alive_t = [t for t in range(T) if alive_mask[:, t].any()]
    if not alive_t:
        return [0]
    if len(alive_t) <= n_frames:
        return alive_t
    sel = np.linspace(0, len(alive_t) - 1, n_frames).round().astype(int)
    return [alive_t[i] for i in sorted(set(sel.tolist()))]


def _analyse(df: pd.DataFrame, filename: str, do_ablation: bool) -> dict:
    _load()
    import tempfile

    # preprocess reuses the exact training-time pipeline
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        df.to_csv(fh, index=False)
        tmp = fh.name
    try:
        X, alive_mask, e_src, e_dst, e_t, index = preprocess_epic_file_sparse(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)

    names = [c for c, _ in sorted(index.cell_to_idx.items(), key=lambda kv: kv[1])]
    data = _build_graph(X, alive_mask, e_src, e_dst, e_t)

    t0 = time.perf_counter()
    logits, _, _ = _MODEL(data.clone())
    probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(1)
    infer_s = time.perf_counter() - t0

    fi_map = build_fate_index()
    inv = {v: k for k, v in fi_map.items()}
    labels, valid = fate_targets(np.array(names, dtype=object), _FATE_MAP, fi_map)

    frames = _pick_frames(alive_mask, N_FRAMES)
    name_to_idx = {n: i for i, n in enumerate(names)}

    cells = []
    for i, nm in enumerate(names):
        alive_f = [fi for fi, t in enumerate(frames) if alive_mask[i, t]]
        pos = [
            [round(float(X[i, 0, t]), 1), round(float(X[i, 1, t]), 1), round(float(X[i, 2, t]), 1)]
            if alive_mask[i, t] else None
            for t in frames
        ]
        sizes = [float(X[i, 3, t]) for t in frames if alive_mask[i, t]]
        parent = lineage_parent(nm)
        rec = {
            "name": nm,
            "fate": inv[int(pred[i])],
            "conf": round(float(probs[i, pred[i]]), 3),
            "size": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
            "b": alive_f[0] if alive_f else 0,
            "p": name_to_idx.get(parent, -1) if parent else -1,
            "pos": pos,
        }
        if valid[i]:
            rec["true_fate"] = inv[int(labels[i])]
            rec["correct"] = bool(pred[i] == labels[i])
        cells.append(rec)

    edges = []
    for t in frames:
        m = e_t == t
        pairs = set()
        for s, d in zip(e_src[m].tolist(), e_dst[m].tolist()):
            if s != d and alive_mask[s, t] and alive_mask[d, t]:
                pairs.add((s, d) if s < d else (d, s))
                if len(pairs) >= MAX_EDGES_PER_FRAME:
                    break
        edges.append([[a, b] for a, b in pairs])

    n_labelled = int(valid.sum())
    accuracy = (
        round(float((pred[valid] == labels[valid]).mean()), 4) if n_labelled else None
    )
    dist = {}
    for c in cells:
        dist[c["fate"]] = dist.get(c["fate"], 0) + 1

    analysis = {
        "n_cells": len(names),
        "n_labelled": n_labelled,
        "accuracy": accuracy,
        "fate_distribution": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
        "mean_confidence": round(float(probs.max(1).mean()), 3),
        "low_confidence_cells": int((probs.max(1) < 0.5).sum()),
        "inference_seconds": round(infer_s, 2),
        "recognised_lineage": n_labelled > 0,
    }

    # graph ablation: does removing contacts change the calls on THIS embryo?
    if do_ablation:
        d2 = _build_graph(X, alive_mask, np.array([]), np.array([]), np.array([]))
        logits2, _, _ = _MODEL(d2)
        pred2 = logits2.argmax(1).numpy()
        analysis["ablation"] = {
            "flip_rate": round(float((pred2 != pred).mean()), 4),
            "accuracy_no_edges": (
                round(float((pred2[valid] == labels[valid]).mean()), 4) if n_labelled else None
            ),
        }

    return {
        "embryo": filename,
        "n_cells": len(names),
        "frames": [int(index.t0 + t) for t in frames],
        "fate_classes": list(FATE_CLASSES),
        "cells": cells,
        "edges": edges,
        "analysis": analysis,
        "metrics": _STATIC_METRICS,
        "explainability": {"saliency": None, "interactions": None},
    }


@app.get("/health")
def health():
    return {"status": "ok", "max_cells": MAX_CELLS}


@app.get("/schema")
def schema():
    return {
        "required_columns": REQUIRED_COLUMNS,
        "format": "EPIC lineage CSV: one row per (cell, timepoint).",
        "notes": [
            "'cell' should use standard C. elegans lineage names (ABa, MSaap, Ea) "
            "so predictions can be scored against the known lineage.",
            "Extra columns are ignored.",
            f"Limit for this hosted demo: {MAX_CELLS} cells.",
        ],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), ablation: bool = True):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB.")

    key = hashlib.sha256(raw).hexdigest()[:16] + f":{ablation}"
    if key in _CACHE:
        return _CACHE[key]

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    _validate(df)
    result = _analyse(df, file.filename or "uploaded.csv", ablation)

    if len(_CACHE) > 12:
        _CACHE.clear()
    _CACHE[key] = result
    return result
