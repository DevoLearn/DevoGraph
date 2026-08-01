# Memo: Fixing the classification task — dataset options

**To:** DevoWorm / DevoGraph mentors
**From:** Barshan Mondal (GSoC — Spatio-Temporal Evolution / ESTGEL)
**Re:** The WT-vs-RNAi classification target and which dataset can actually support it

---

## 1. The problem I found

While preparing the explainability deliverable, I traced the training results back and found the
classification task, as currently set up on the EPIC data, is **not learning a real signal**.

- **Labels are degenerate:** `labels.csv` is **254 "perturbed/RNAi" vs 6 "control"**.
- **Reported accuracy is the base rate:** val accuracy sits at exactly **0.9615 = 25/26** every epoch,
  and train accuracy collapses to the identical value. The model simply predicts the majority
  class for every embryo — it has learned nothing discriminative.
- **The labels are also biologically wrong:** the EPIC files are almost all **wild-type embryos
  carrying different fluorescent *reporter* strains** (`pha4`, `cnd1`, `tbx-8`, `nhr-25`, `dyf7red`, …).
  The `blot` fluorescence *is* the reporter. Our heuristic tagged every strain without a
  `bright`/`norfp` keyword as "RNAi-perturbed," which mislabels ~254 phenotypically wild-type
  embryos.

**Root cause:** the standard EPIC lineage set does not naturally contain a wild-type-vs-RNAi split,
which is the graph-classification target the proposal committed to. The *pipeline* (tensor masking,
proximity + lineage graph init, EAM / DRL / DNL, attention export) is sound and is the real
contribution — but the discriminative task on top of it needs a dataset that has genuine
perturbation labels.

## 2. Datasets that could provide a real signal

| Dataset | Perturbation labels | Depth | Per-cell data | Access |
|---|---|---|---|---|
| **RIKEN WDDD / SSBD "Kyoda-WormEmbryoRNAi"** | **33 WT + 1142 RNAi**, 263 genes (≥5 embryos for 189 genes) | first ~3 divisions (→ ~8-cell) | 3D nuclear coords, outlines, division timing | Open (CC BY-SA), BD5 / BDML / ome.zarr |
| **CShaper 4D Morphological Atlas** (Nat Commun 2020) | Mostly WT (17 embryos) | 4 → 350-cell | cell shape, volume, surface area, migration, nucleus pos, **cell–cell contact** | Supplementary + repo |
| **System-Level Morphogenesis Phenotyping** (bioRxiv 776062) | **222 WT + 758-gene RNAi** | 4 → 350-cell, 226 cells | migration trajectories, cell arrangement | availability to confirm |
| **Digital Development DB** | 204-gene lineage phenotype annotations | lineage-level | curated phenotype flags | web DB (label source) |
| **EPIC (current)** | ~all WT reporter strains — no real split | 350-cell, has `blot` expression | x, y, z, size, blot | already downloaded |

## 3. My recommendation

**Primary: adopt the RIKEN WDDD / SSBD Kyoda RNAi collection for the classification + explainability.**
It was purpose-built as wild-type vs. RNAi, has hundreds of labeled embryos, is openly downloadable,
and its per-nucleus 3D coordinates + division timing map directly onto our existing `X ∈ (N, d, T)`
tensor and proximity/lineage graph init. It converts the fake 96% into a real, balanced task and even
offers a **multi-class (per-gene) target** for richer explainability. Its one cost is developmental
depth: only the first ~3 divisions, so smaller graphs and a weaker "growing-node / multiscale" story.

**Complement: CShaper for depth and better edges.** CShaper provides true **cell–cell contact maps to
the 350-cell stage** — far better graph edges than our current Euclidean-distance threshold — and keeps
the deep-lineage multiscale narrative alive. Mostly WT, so it complements rather than replaces the
labels from WDDD.

**Strongest package = hybrid:** WDDD/SSBD for labeled WT-vs-RNAi classification + explainability;
CShaper contact graphs to enrich edge construction and the multiscale story.

## 4. Questions for the mentors

1. Is switching the classification target to the **WDDD/SSBD RNAi** collection acceptable, given the
   shallower developmental depth (~8-cell)?
2. Do you prefer the **hybrid** (WDDD labels + CShaper 350-cell contact graphs), accepting the extra
   integration work?
3. Or should EPIC stay primary with the task **reframed** away from WT-vs-RNAi (e.g. a self-supervised
   attention-over-development objective), transparently noting the label limitation?
4. Any known internal DevoWorm curation of genuine EPIC RNAi/mutant embryos I should use instead?

## 5. Sources

- EPIC dataset — https://epic.gs.washington.edu/
- WDDD2 (browsable) — https://wddd.riken.jp/yawddd/
- SSBD "2-Kyoda-WormEmbryoRNAi" — https://ssbd.riken.jp/database/2/
- Kyoda et al., *Deep Collection of Quantitative Nuclear Division Dynamics Data in RNAi-treated C. elegans Embryos* — https://www.biorxiv.org/content/10.1101/2020.10.04.325761v1
- CShaper morphological atlas (Nat Commun 2020) — https://www.nature.com/articles/s41467-020-19863-x
- System-level morphogenesis phenotyping (bioRxiv 776062) — https://www.biorxiv.org/content/10.1101/776062v1
- Digital Development database (NAR 2016) — https://academic.oup.com/nar/article/44/D1/D781/2502623
