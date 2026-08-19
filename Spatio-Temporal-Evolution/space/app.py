"""
ESTGEL cell-fate demo, Gradio Space.

Upload an EPIC-format lineage CSV of a C. elegans embryo and the trained ESTGEL
node classifier predicts the terminal tissue fate of every cell, then reports how
much those calls depended on the cell-cell contact graph.

This is the free-tier entrypoint (Gradio SDK). The same inference code also backs a
REST endpoint in service.py, which the dashboard at barshan.is-a.dev calls.

The model is small and CPU-bound, so nothing here needs a GPU. ZeroGPU requires at
least one decorated function to exist, so a no-op is declared and never used for the
real work, which means no GPU quota is consumed.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # only present on Hugging Face infrastructure
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():
        """Satisfies the ZeroGPU requirement for a decorated function. Never called."""
        return None
except Exception:  # pragma: no cover - local runs
    pass

from service import MAX_CELLS, MIN_CELLS, REQUIRED_COLUMNS, _analyse  # noqa: E402

ROOT = Path(__file__).resolve().parent

FATE_COLORS = {
    "neuron": "#f4b942", "muscle": "#ef5a5a", "hypodermis": "#2fc3b4",
    "pharynx": "#b57be8", "intestine": "#6cc24a", "glia": "#4d9de0",
    "germline": "#e879c0", "excretory": "#d99a4e", "coelomocyte": "#9aa7bd",
    "death": "#5b6472", "other": "#7a8699", "mixed": "#3d4757", "unknown": "#3d4757",
}
BG, INK, MUTED = "#0a0d13", "#e8eef7", "#93a0b4"


def _project(P, yaw=0.6, pitch=-0.35):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    x1 = X * cy + Z * sy
    z1 = -X * sy + Z * cy
    return x1, Y * cp - z1 * sp, Y * sp + z1 * cp


def _plot(result: dict):
    """Final-frame view of the embryo, cells coloured by predicted fate."""
    cells, frames = result["cells"], result["frames"]
    fi = len(frames) - 1
    pts, fates, sizes, miss = [], [], [], []
    for c in cells:
        p = c["pos"][fi]
        if p is None:
            continue
        pts.append(p); fates.append(c["fate"]); sizes.append(c["size"])
        miss.append(c.get("correct") is False)
    if not pts:
        return None
    P = np.array(pts, float)
    x, y, z = _project(P - P.mean(axis=0))
    sizes, miss = np.array(sizes), np.array(miss)
    order = np.argsort(z)
    depth = (z - z.min()) / (np.ptp(z) or 1)

    lim_x, lim_y = 1.08 * np.abs(x).max(), 1.12 * np.abs(y).max()
    fig, ax = plt.subplots(figsize=(11, 11 * float(lim_y / lim_x) + 1.0), dpi=130, facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(x[order], y[order],
               s=(14 + sizes[order] * 1.2) * (0.45 + 0.9 * depth[order]),
               c=[FATE_COLORS.get(fates[i], "#7a8699") for i in order],
               alpha=0.95, linewidths=0)
    if miss.any():
        ax.scatter(x[miss], y[miss], s=(30 + sizes[miss] * 1.4), facecolors="none",
                   edgecolors="#e56a6a", linewidths=1.1, alpha=0.85)
    from matplotlib.lines import Line2D
    present = sorted({f for f in fates}, key=lambda f: -fates.count(f))
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markeredgecolor="none",
                      markerfacecolor=FATE_COLORS.get(f, "#7a8699"), label=f) for f in present]
    if miss.any():
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=9,
                              markerfacecolor="none", markeredgecolor="#e56a6a",
                              label="misclassified"))
    ax.legend(handles=handles, loc="upper right", frameon=False, labelcolor=MUTED, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y)
    ax.set_title(f"Predicted cell fate  ·  t = {frames[fi]} min  ·  {len(P)} cells",
                 color=INK, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def _summary(a: dict) -> str:
    lines = [f"**{a['n_cells']} cells analysed** in {a['inference_seconds']}s on CPU.", ""]
    if a["accuracy"] is not None:
        lines.append(
            f"- **Accuracy {a['accuracy']*100:.1f}%** on {a['n_labelled']} cells whose "
            f"fate is known from the C. elegans lineage"
        )
    else:
        lines.append(
            "- Cell names are not standard lineage names, so there is no ground truth "
            "to score against. Predictions only."
        )
    lines.append(f"- Mean confidence {a['mean_confidence']*100:.0f}%, "
                 f"{a['low_confidence_cells']} cells below 50%")
    if a.get("ablation"):
        lines.append(
            f"- Removing every cell-cell contact flips "
            f"**{a['ablation']['flip_rate']*100:.0f}% of the predictions**, so the contact "
            f"graph is carrying real information about fate"
        )
    lines.append("")
    lines.append("Predicted tissue distribution: " +
                 ", ".join(f"{k} {v}" for k, v in a["fate_distribution"].items()))
    return "\n".join(lines)


def analyse(file_obj):
    if file_obj is None:
        return "Upload an EPIC-format CSV to begin.", None, None, None
    try:
        df = pd.read_csv(file_obj.name)
    except Exception as exc:
        return f"Could not read that CSV: {exc}", None, None, None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return (f"CSV is missing required column(s): {', '.join(missing)}. "
                f"Expected: {', '.join(REQUIRED_COLUMNS)}."), None, None, None
    n = df["cell"].astype(str).nunique()
    if n < MIN_CELLS:
        return (f"Only {n} distinct cells. The edge-attention module splits the graph into "
                f"11 nested subgraphs, so at least {MIN_CELLS} cells are needed."), None, None, None
    if n > MAX_CELLS:
        return (f"{n} cells is above the {MAX_CELLS} limit for this hosted demo. "
                f"Run the model locally for larger embryos."), None, None, None

    result = _analyse(df, Path(file_obj.name).name, True)

    rows = [
        {
            "cell": c["name"],
            "predicted fate": c["fate"],
            "confidence": c["conf"],
            "known fate": c.get("true_fate", ""),
            "correct": "" if "correct" not in c else ("yes" if c["correct"] else "no"),
        }
        for c in result["cells"]
    ]
    table = pd.DataFrame(rows).sort_values("confidence", ascending=False)

    out = Path(tempfile.gettempdir()) / "estgel_predictions.json"
    out.write_text(json.dumps(result), encoding="utf-8")

    return _summary(result["analysis"]), _plot(result), table, str(out)


DESCRIPTION = """
# Cell-fate prediction in *C. elegans* embryos

Upload a lineage CSV and this predicts what tissue each cell becomes: neuron, muscle,
hypodermis, pharynx, intestine and so on.

The model is **ESTGEL**, a spatio-temporal graph network. It sees each cell's 3D position,
size, expression and its contacts with neighbouring cells over developmental time. It never
sees the cell's *name*, because in *C. elegans* the name determines the fate, so using it
would be trivial label leakage.

Built for [DevoWorm / DevoGraph](https://github.com/DevoLearn/DevoGraph) during Google Summer of Code.
"""

CSV_HELP = f"""
**Format.** One row per cell per timepoint, with at least: `{', '.join(REQUIRED_COLUMNS)}`.
Extra columns are ignored. This is the [EPIC](https://epic.gs.washington.edu/) lineage format.

Use standard lineage names (`ABa`, `MSaap`, `Ea`) and the predictions are also scored against
the known lineage. With other names you still get predictions, just no accuracy.

Limits: {MIN_CELLS} to {MAX_CELLS} cells.
"""

with gr.Blocks(title="ESTGEL cell-fate", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Embryo CSV", file_types=[".csv"])
            run = gr.Button("Predict cell fates", variant="primary")
            gr.Markdown(CSV_HELP)
        with gr.Column(scale=2):
            summary = gr.Markdown("Upload an EPIC-format CSV to begin.")
            plot = gr.Plot(label="Predicted fate")
    with gr.Row():
        table = gr.Dataframe(label="Per-cell predictions", wrap=True, max_height=420)
    download = gr.File(label="Full results (JSON)")

    examples = ROOT / "examples"
    if examples.is_dir():
        sample = sorted(examples.glob("*.csv"))
        if sample:
            gr.Examples(examples=[[str(p)] for p in sample], inputs=[file_in],
                        label="Example embryo")

    run.click(analyse, inputs=[file_in], outputs=[summary, plot, table, download])

if __name__ == "__main__":
    demo.launch()
