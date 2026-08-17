# DHGNN — Dynamic Hypergraph Neural Network for C. elegans Embryogenesis

**Open Source Cohort / DevoWorm Project 2026**

---

## Project Overview

This project extends the DevoGraph framework by building a **Dynamic Hypergraph Neural Network (DHGNN)** to model C. elegans embryogenesis. It integrates three biologically distinct hyperedge types into a unified incidence matrix and learns per-cell embeddings across developmental time. The model is trained on **cell fate** (founder-lineage classification) and evaluated by fate accuracy and by the **lineage-coherence** of the learned embeddings.

### Three Hyperedge Types

| Type | Source | Fixed? | Biological Meaning |
|---|---|---|---|
| **Spatial** | `ce_temporal_data.csv` + DBSCAN | Rebuilt each t | Cells in physical proximity at timepoint t |
| **Lineage** | `cells_birth_and_pos.csv` | Fixed | Parent–daughter division bonds |
| **Functional** | `Connectome.csv` + `Alignment_map_csv.csv` | Fixed | Adult FFL neural circuit motifs |


---
