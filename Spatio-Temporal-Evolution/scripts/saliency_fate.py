"""
Causal saliency / ablation explainability for the ESTGEL fate model.

The EAM attention is near-saturated, so instead of reading attention magnitudes we
PERTURB inputs and measure how the predicted-fate probability responds:

  1. Feature occlusion  - mean-impute each input feature (x,y,z,size,blot) and measure
     the drop in P(predicted fate) per cell -> which features drive each tissue.
  2. Graph ablation      - remove ALL cell-cell edges and re-predict -> does the model
     actually use interactions for fate, or is it position alone?
  3. Gradient saliency   - |d logit_pred / d feature| as an independent cross-check.

Aggregated over several embryos. Outputs to checkpoints/estgel_fate/:
  fate_saliency.json, fate_saliency.png
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cell_fate import FATE_CLASSES, build_fate_index, fate_targets, load_fate_csv
from src.epic_dataset import EpicEmbryoDataset
from src.estgel_node import ESTGELNodeClassifier

FEATURES = ["x", "y", "z", "size", "blot"]


def load_model(cfg, device):
    drl = {}
    if cfg.get("drl_channels"):
        drl = {"Lz": cfg["drl_channels"], "Lr": cfg["drl_channels"], "Lh": cfg["drl_channels"]}
    m = ESTGELNodeClassifier(
        num_classes=len(FATE_CLASSES), K=cfg["K"], in_dim=5,
        recurrence_stride=cfg["recurrence_stride"], max_steps=cfg["max_steps"],
        bptt_truncation=cfg["bptt_truncation"], max_nodes=cfg["max_nodes"], **drl,
    ).to(device)
    m.load_state_dict(torch.load(REPO_ROOT / "checkpoints" / "estgel_fate" / "best.pt",
                                 map_location=device)["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def predict_probs(model, data):
    logits, _, _ = model(data.clone())
    return torch.softmax(logits, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-embryos", type=int, default=6)
    ap.add_argument("--max-cells", type=int, default=1000)
    args = ap.parse_args()

    ckpt = REPO_ROOT / "checkpoints" / "estgel_fate"
    cfg = json.loads((ckpt / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fi = build_fate_index(); inv = {v: k for k, v in fi.items()}
    cell_fate = load_fate_csv()
    model = load_model(cfg, device)

    ds = EpicEmbryoDataset(cfg["processed_dir"], use_global_index=False)
    # pick small embryos for tractable repeated forwards
    sizes = [(i, len(np.load(ds.processed_dir / ds.filenames[i], allow_pickle=True)["idx_to_cell"]))
             for i in range(len(ds))]
    chosen = [i for i, n in sorted(sizes, key=lambda x: x[1]) if n <= args.max_cells][:args.n_embryos]

    occ = defaultdict(lambda: np.zeros(len(FEATURES)))  # fate -> summed conf-drop per feature
    occ_n = defaultdict(int)
    grad_imp = defaultdict(lambda: np.zeros(len(FEATURES)))
    grad_n = defaultdict(int)
    # graph ablation accumulators
    abl_drop = defaultdict(float); abl_n = defaultdict(int); flips = 0; flip_tot = 0
    acc_full = acc_noedge = acc_tot = 0

    for idx in chosen:
        data0 = ds[idx]
        labels, valid = fate_targets(data0.idx_to_cell, cell_fate, fi)
        P0 = predict_probs(model, data0).cpu().numpy()
        pred = P0.argmax(1)
        rows = np.arange(len(pred))
        p_pred = P0[rows, pred]

        # accuracy full
        vm = valid
        acc_full += int((pred[vm] == labels[vm]).sum()); acc_tot += int(vm.sum())

        # ---- feature occlusion (set feature to its train mean => normalized 0) ----
        fmean = model.feat_mean.cpu().numpy()
        for k in range(len(FEATURES)):
            d = ds[idx]
            d.x[:, k, :] = float(fmean[k])
            Pk = predict_probs(model, d).cpu().numpy()
            drop = p_pred - Pk[rows, pred]           # confidence lost when feature removed
            for i in range(len(pred)):
                f = inv[int(pred[i])]
                occ[f][k] += float(drop[i]);
            for i in range(len(pred)):
                occ_n[inv[int(pred[i])]] += 1 if k == 0 else 0
        # normalize occ_n counts (added once per node above only at k==0)

        # ---- graph ablation: remove all edges ----
        d = ds[idx]
        d.edge_index = torch.empty((2, 0), dtype=torch.long)
        d.edge_t = torch.empty((0,), dtype=torch.long)
        Pne = predict_probs(model, d).cpu().numpy()
        pred_ne = Pne.argmax(1)
        drop = p_pred - Pne[rows, pred]
        for i in range(len(pred)):
            abl_drop[inv[int(pred[i])]] += float(drop[i]); abl_n[inv[int(pred[i])]] += 1
        flips += int((pred_ne != pred).sum()); flip_tot += len(pred)
        acc_noedge += int((pred_ne[vm] == labels[vm]).sum())

        # ---- gradient saliency ----
        try:
            d = ds[idx]
            x = d.x.clone().to(device).requires_grad_(True)
            d.x = x
            logits, _, _ = model(d)
            sel = logits[rows, pred]                  # predicted-class logit per node
            sel.sum().backward()
            g = x.grad.abs().to(device)               # (N,d,T)
            am = d.alive_mask.to(device).float()
            gpf = (g * am.unsqueeze(1)).sum(2) / am.sum(1, keepdim=True).clamp(min=1)  # (N,d)
            gpf = gpf.detach().cpu().numpy()
            for i in range(len(pred)):
                f = inv[int(pred[i])]
                grad_imp[f] += gpf[i]; grad_n[f] += 1
        except Exception as e:
            print("grad saliency skipped:", e)

    # aggregate
    def norm_dict(dsum, dcnt):
        return {f: (dsum[f] / max(dcnt[f], 1)).tolist() if hasattr(dsum[f], 'tolist')
                else dsum[f] / max(dcnt[f], 1) for f in dsum}

    occ_mean = {f: (occ[f] / max(occ_n[f], 1)).round(4).tolist() for f in occ}
    grad_mean = {f: (grad_imp[f] / max(grad_n[f], 1)).round(4).tolist() for f in grad_imp}
    abl_mean = {f: round(abl_drop[f] / max(abl_n[f], 1), 4) for f in abl_drop}

    report = {
        "embryos_used": [ds.filenames[i] for i in chosen],
        "features": FEATURES,
        "accuracy_full": round(acc_full / max(acc_tot, 1), 4),
        "accuracy_no_edges": round(acc_noedge / max(acc_tot, 1), 4),
        "prediction_flip_rate_no_edges": round(flips / max(flip_tot, 1), 4),
        "feature_occlusion_confdrop_by_fate": occ_mean,
        "graph_ablation_confdrop_by_fate": abl_mean,
        "gradient_saliency_by_fate": grad_mean,
    }
    (ckpt / "fate_saliency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- figure ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fates = [f for f in ["neuron", "muscle", "hypodermis", "pharynx", "intestine"] if f in occ_mean]
        M = np.array([occ_mean[f] for f in fates])          # (F_fates, 5 features)
        fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=140,
                               gridspec_kw={"width_ratios": [3, 1]})
        im = ax[0].imshow(M, cmap="RdBu_r", aspect="auto",
                          vmin=-np.abs(M).max(), vmax=np.abs(M).max())
        ax[0].set_xticks(range(len(FEATURES))); ax[0].set_xticklabels(FEATURES)
        ax[0].set_yticks(range(len(fates))); ax[0].set_yticklabels(fates)
        ax[0].set_title("Feature occlusion: confidence drop when feature removed\n(red = feature supports the fate)",
                        fontsize=11, fontweight="bold")
        for i in range(len(fates)):
            for j in range(len(FEATURES)):
                ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

        gfates = [f for f in fates if f in abl_mean]
        ax[1].barh(gfates, [abl_mean[f] for f in gfates], color="#c0504d")
        ax[1].set_title("Graph ablation:\nconf drop with NO edges", fontsize=11, fontweight="bold")
        ax[1].axvline(0, color="k", lw=0.8)
        fig.suptitle(f"ESTGEL fate saliency  |  acc full {report['accuracy_full']:.2f} -> "
                     f"no-edges {report['accuracy_no_edges']:.2f}  (flip {report['prediction_flip_rate_no_edges']:.0%})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(); fig.savefig(ckpt / "fate_saliency.png", bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)

    print(f"Embryos: {len(chosen)} | acc full {report['accuracy_full']:.3f} -> "
          f"no-edges {report['accuracy_no_edges']:.3f} | flip rate {report['prediction_flip_rate_no_edges']:.3f}")
    print("\nFeature occlusion (conf drop when removed) by fate:")
    print(f"  {'fate':11}" + "".join(f"{f:>8}" for f in FEATURES))
    for f in ["neuron", "muscle", "hypodermis", "pharynx", "intestine"]:
        if f in occ_mean:
            print(f"  {f:11}" + "".join(f"{v:>8.3f}" for v in occ_mean[f]))
    print("\nGraph-ablation conf drop by fate:", abl_mean)
    print("Saved fate_saliency.json, fate_saliency.png")


if __name__ == "__main__":
    main()
