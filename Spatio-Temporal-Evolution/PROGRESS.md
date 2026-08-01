# Progress notes — Cell-fate reframe

_Spatio-Temporal-Evolution (ESTGEL / DevoGraph GSoC) — 2026-07-23_

## 1. Problem found
The original **WT-vs-RNAi graph classification** was degenerate:
- `labels.csv` was **254 "RNAi" vs 6 "control"**; val accuracy sat at exactly **0.9615 = 25/26**
  (the majority base rate) — the model just predicted one class.
- EPIC files are almost all **wild-type reporter strains**, not RNAi mutants, so the keyword
  heuristic mislabeled ~254 WT embryos. EPIC has no real WT-vs-RNAi split.

## 2. Dataset research
Surveyed DevoWorm / SSBD / literature for data with a real signal (see `DATASET_MEMO.md`):
- **RIKEN WDDD/SSBD RNAi** — 33 WT + 1142 RNAi, open BD5/BDML (~30 MB/embryo quantitative). Best
  for a genuine perturbation split; shallow (~8-cell).
- **CShaper atlas** — cell–cell contact maps to 350-cell (better edges; mostly WT).
- **Digital Development DB** — ~1400 perturbed lineages with fate phenotypes (mutation dimension).

## 3. Reframed target: cell-fate prediction
The real goal is **predicting each cell's terminal fate from its division pattern + spatio-temporal
context** — a **node-level** task, not whole-graph. Because the C. elegans lineage is invariant,
fate labels come free from the cell name (Sulston), so **EPIC (already downloaded) is the right
feature set**. Design rule: the **cell name is never a model feature** (that would be trivial
leakage); fate is predicted from position / timing / contacts / expression only.

## 4. Built this session (smoke-tested end-to-end on GPU)
| File | What |
|---|---|
| `src/cell_fate.py` | 13-class fate taxonomy + `CellFateMap` (terminal desc → class; progenitor → consensus of descendant leaves w/ purity; `mixed` if uncommitted) + `fate_targets()` |
| `scripts/build_cell_fate.py` | Generates the label map |
| `Dataset/fate/cell_fate.csv` | **2,769 cells labeled**, 2 unknown; source `.xlsx` staged |
| `src/estgel_node.py` | `ESTGELNodeClassifier` — reuses EAM→DRL→DNL, per-node embedding from `X_hat` → fate |
| `scripts/train_fate.py` | By-embryo split, class-weighted loss, reports acc + **macro-F1 + per-class recall** |

Label map source: DevoWorm `name-function-each-cell` (WormAtlas terminal descriptions),
693/710 EPIC cells reconcile directly; occurrence-weighted distribution is healthy multi-class
(neuron 18%, pharynx 16%, hypodermis 15%, muscle 12%, glia/intestine 5%, `mixed` 26%).

## 5. Trained fate model (result)
After fixing two real bugs — **no feature normalization** (blot ranged to 1.6e5 → exploding logits,
loss 467; also explains why the original graph classifier never learned) and a **temporal-coverage
bug** (fixed early window left ~70% of late-born cells with zero embeddings → all-"neuron" predictor)
— the node classifier trains cleanly:

- **Val accuracy 0.635, macro-F1 0.343** over 35 epochs (15 + 20 more with cosine LR decay;
  was a fake plateau of 0.25 / 0.05 before the fixes). Plateaued — the remaining errors are the
  genuinely hard cases, not a training-time issue.
- Per-class recall: pharynx 0.80, hypodermis 0.72, muscle 0.72, neuron 0.62, intestine 0.60;
  rare classes (glia/germline/excretory/coelomocyte) ~0 (dragged by <1% frequency; unweighted).
- Checkpoint: `checkpoints/estgel_fate/best.pt` (feature-normalization stats saved in buffers).
- Design rule kept: cell NAME is never a feature; fate predicted from position/size/expression + graph.

## 6. Explainability (saliency / ablation, not attention)
EAM attention is near-saturated (~0.9-1.0), so causal perturbation is used instead
(`scripts/extract_fate_explainability.py`, `scripts/saliency_fate.py`):

- **Interactions causally matter (validates ESTGEL):** removing ALL cell-cell edges drops accuracy
  0.46 → 0.31 and **flips 40% of fate predictions**. Most edge-dependent: hypodermis, pharynx, neuron;
  least: muscle, intestine.
- **Feature importance by tissue:** position (x,y,z) dominates fate; **size** distinctively drives
  intestine and pharynx; **blot (reporter expression) ≈ 0** — honest: the model uses geometry, not the
  per-embryo reporter.
- **Contact structure** (fate-annotated) is biologically sensible: like-fate cells cluster; intestine
  is isolated from neuron; AB-derived pharynx/neuron/hypodermis intermix.
- Artifacts: `fate_interaction_map.png`, `fate_saliency.png`, `fate_explainability.json`, `fate_saliency.json`.

## 6b. Interactive dashboard / developmental simulation
`dashboard/index.html` — self-contained (no CDNs, data inlined). Hand-rolled canvas 3D:
rotatable embryo colored by predicted fate that **plays development forward** — smooth position
interpolation across 30 frames, **cells divide** (daughters emerge from parents via lineage),
**contact edges** drawn per frame, brightness by prediction confidence, red rings on misclassified
cells (toggleable). Click-to-inspect (fate, confidence, true fate, parent, birth). Plus the
causal-explainability panels (graph ablation, feature occlusion, contact map, training curve, per-class recall).
Regenerate for any embryo: `python scripts/export_dashboard_data.py --embryo <name>.npz`
(rebuilds the data + injects into `dashboard/fate_dashboard.template.html` -> `index.html`).
The old WT/RNAi/attention dashboard (app.js/style.css/embryo_data.json/explainability_report.json)
was retired.

## 7. Next steps
- Lift rare classes (√-capped class weighting already in `train_fate.py`, drop `--no-class-weight`).
- Wire per-cell predicted fate + interactions into the 3D `dashboard/`.
- Bring in Digital Development / WDDD for the **mutation → fate change** dimension.
- Open decision (see `DATASET_MEMO.md`): confirm dataset direction with mentors.
