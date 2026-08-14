---
title: ESTGEL Cell Fate API
emoji: 🪱
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ESTGEL cell-fate API

Predicts the terminal tissue fate of every cell in a *C. elegans* embryo from an
EPIC-format lineage CSV, using the trained ESTGEL node classifier.

This backend exists so a static site can offer CSV upload: the dashboard at
`barshan.is-a.dev/estgel/` posts a file here and renders the JSON it gets back.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness check |
| `GET` | `/schema` | required CSV columns and limits |
| `POST` | `/predict` | `multipart/form-data` with `file=<csv>`, optional `?ablation=false` |

### Example

```bash
curl -X POST https://<your-space>.hf.space/predict -F "file=@embryo.csv"
```

### CSV format

One row per cell per timepoint, with at least:

```
cell, time, x, y, z, size, blot
```

Extra columns are ignored. Use standard lineage names (`ABa`, `MSaap`, `Ea`) and the
response also scores predictions against the known *C. elegans* lineage. With
non-standard names you still get predictions, just no accuracy.

Limits: 12 to 1200 cells, 25 MB. The lower bound exists because the edge-attention
module decomposes the graph into 11 nested subgraphs.

### Response

Matches the dashboard's data format (`embryo`, `frames`, `cells`, `edges`, `metrics`)
plus an `analysis` block:

```json
{
  "analysis": {
    "n_cells": 710, "n_labelled": 517, "accuracy": 0.7582,
    "fate_distribution": {"neuron": 228, "pharynx": 173, "...": 0},
    "mean_confidence": 0.65, "low_confidence_cells": 146,
    "inference_seconds": 1.24, "recognised_lineage": true,
    "ablation": {"flip_rate": 0.6197, "accuracy_no_edges": 0.3056}
  }
}
```

`ablation` re-runs the model with every cell-cell edge removed, so `flip_rate` is how
many fate calls depended on the contact graph for that specific embryo.

## Deploying

The Space needs the source modules, the checkpoint and the fate labels next to
`app.py`. Assemble them from the repo:

```bash
python scripts/build_api_bundle.py
```

That fills `api/src/` and `api/model/`. Then create a Space (Docker SDK) and push the
contents of `api/`:

```bash
git clone https://huggingface.co/spaces/<user>/estgel-cell-fate
cp -r api/* estgel-cell-fate/
cd estgel-cell-fate && git add . && git commit -m "Deploy ESTGEL API" && git push
```

Model weights are about 320 KB, so the whole bundle is small enough for a free Space.

## Connecting the dashboard

Set the API base before the dashboard script runs:

```html
<script>window.ESTGEL_API = "https://<your-space>.hf.space";</script>
```

Without it the upload control stays disabled and the dashboard still works as a
static viewer of the bundled embryo.

CORS currently allows `barshan.is-a.dev`, `blueee04.github.io` and localhost. Add
origins in `app.py` if you host it elsewhere.

## Notes

* CPU inference is roughly 1 to 4 seconds for a 700-cell embryo. Free Spaces sleep
  when idle, so the first request after a quiet period takes about 30 seconds to wake.
* Responses are cached by file hash, so re-uploading the same CSV returns instantly.
* The cell name is never a model input. Fate is predicted from position, size,
  expression and graph structure, and names are used only to look up ground truth.
