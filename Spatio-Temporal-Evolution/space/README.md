---
title: ESTGEL Cell Fate
emoji: 🪱
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Predict C. elegans cell fate from an embryo lineage CSV
---

# ESTGEL cell-fate demo

Upload an EPIC-format lineage CSV of a *C. elegans* embryo. The trained ESTGEL
spatio-temporal graph network predicts the terminal tissue fate of every cell, scores
those predictions against the known lineage where the names allow it, and reports how
many calls depended on the cell-cell contact graph.

Built for [DevoWorm / DevoGraph](https://github.com/DevoLearn/DevoGraph) during Google
Summer of Code.

## Hardware

The model is 80k parameters and runs in a few seconds on CPU, so no GPU is needed. On a
free account this is hosted on `zero-a10g` because that is the only free tier that allows
a Gradio Space; a no-op decorated function satisfies the ZeroGPU requirement and the real
work stays outside it, so no GPU quota is consumed.

## Also a REST API

`service.py` exposes the same inference over HTTP for the dashboard at
`barshan.is-a.dev/estgel/`. To run it standalone:

```bash
uvicorn service:app --port 7860
```

## Notes

The cell name is never a model input. Fate is predicted from 3D position, size,
expression and graph structure over developmental time; names are used only to look up
ground truth for scoring.
