"""
Cell-fate labels for C. elegans embryo cells (node-level classification targets).

Because the C. elegans lineage is invariant, every named cell has a known terminal
fate. This module derives a coarse tissue/fate class for any lineage-named cell
(e.g. "ABalaaaa", "MSaap", "Ea") from the canonical WormAtlas terminal-cell
descriptions distributed with DevoWorm (``name-function-each-cell``).

Two kinds of cells:

* **Terminal cells** — an exact lineage name in the reference; class comes straight
  from its free-text description via keyword rules (:data:`FATE_RULES`).
* **Progenitors** — the internal nodes actually present in EPIC embryos (which trace
  only to ~350 cells). A progenitor's fate is the **consensus** of the terminal
  fates of all its lineage descendants (leaves whose normalized lineage name has the
  progenitor's name as a prefix). ``purity`` records the fraction that agree; cells
  below :data:`MIXED_PURITY` are labeled ``mixed`` rather than forced into one class.

IMPORTANT (training): fate is a deterministic function of the *cell name*. Do NOT
feed the name (or a name-derived id) as a model feature — that is trivial label
leakage. Predict fate from spatio-temporal signal (position, timing, contacts,
expression) only; use these labels solely as targets.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = REPO_ROOT / "Dataset" / "fate" / "name-function-each-cell.xlsx"
DEFAULT_SHEET = "name-function-each-cell"

# Coarse fate taxonomy. `mixed` and `unknown` are assigned by the resolver, not by
# a description rule, so they are listed separately from FATE_CLASSES-by-keyword.
FATE_CLASSES: tuple[str, ...] = (
    "neuron",
    "muscle",
    "hypodermis",
    "pharynx",
    "intestine",
    "glia",
    "germline",
    "excretory",
    "coelomocyte",
    "death",
    "other",
    "mixed",
    "unknown",
)

# Ordered (first match wins) keyword rules over the lowercased description.
# Order matters: e.g. "pharyngeal muscle" must resolve to pharynx, not muscle;
# sheath/socket (glia) must win over the "neuron" it is often described next to.
FATE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("death", ("programmed cell death", "cell death", "apoptot", "(dies")),
    ("germline", ("germ line", "germline", "germ cell", "gonad", "distal tip",
                   "sperm", "oocyte", "spermathec")),
    ("intestine", ("intestin", "gut")),
    ("pharynx", ("pharyn", "marginal cell", "pharyngeal")),
    ("glia", ("sheath", "socket")),
    ("muscle", ("muscle", "body wall", "body-wall")),
    ("hypodermis", ("hypoderm", "hyp7", "seam", "epiderm", "syncyti", "tail spike")),
    ("excretory", ("excretory",)),
    ("coelomocyte", ("coelom",)),
    ("neuron", ("neuron", "interneuron", "sensory", "motor", "ganglion",
                 "nerve", "dopaminerg", "cholinerg", "amphid", "labial",
                 "cephalic", "ray")),
)

MIXED_PURITY = 0.6  # below this fraction of agreeing descendants -> "mixed"

# Founder / early cells not resolvable from terminal leaves (they die out or are
# germline precursors). Explicit fallbacks keyed on exact name.
FOUNDER_FATE: dict[str, str] = {
    "P4": "germline", "Z2": "germline", "Z3": "germline",
    "P0": "other", "P1": "mixed", "P2": "mixed", "P3": "mixed",
    "EMS": "mixed", "AB": "mixed", "MS": "mixed",
}


def _norm(name: object) -> str:
    """Normalize a lineage name: strip whitespace (e.g. 'AB plapaaaapp' -> 'ABplapaaaapp')."""
    return re.sub(r"\s+", "", str(name))


def classify_description(description: object) -> str:
    """Map a free-text terminal-cell description to a coarse fate class."""
    text = str(description).lower()
    for fate, keywords in FATE_RULES:
        if any(k in text for k in keywords):
            return fate
    return "other"


@dataclass(frozen=True)
class FateResolution:
    fate: str
    method: str      # "founder" | "exact" | "consensus" | "ancestor" | "unknown"
    purity: float    # 1.0 for exact/founder; consensus fraction otherwise
    n_leaves: int    # number of terminal descendants considered


class CellFateMap:
    """Resolves any lineage-named cell to a coarse fate class."""

    def __init__(self, leaf_fate: dict[str, str]) -> None:
        # leaf_fate: normalized terminal lineage name -> fate class
        self.leaf_fate = leaf_fate
        self._leaf_items = sorted(leaf_fate.items())

    @classmethod
    def from_reference(
        cls,
        reference_path: str | Path = DEFAULT_REFERENCE,
        sheet: str = DEFAULT_SHEET,
    ) -> "CellFateMap":
        df = pd.read_excel(reference_path, sheet_name=sheet).dropna(subset=["Cell"])
        leaf_fate: dict[str, str] = {}
        for _, row in df.iterrows():
            lin = _norm(row["Lineage Name"])
            if not lin or "." in lin:
                # skip post-embryonic sublineage names (e.g. "P9.aapa"); EPIC is embryonic
                continue
            leaf_fate[lin] = classify_description(row.get("Description", ""))
        return cls(leaf_fate)

    def _descendant_leaves(self, name: str) -> list[str]:
        return [fate for leaf, fate in self._leaf_items if leaf.startswith(name)]

    def resolve(self, cell: str) -> FateResolution:
        name = _norm(cell)
        if name in FOUNDER_FATE and name not in self.leaf_fate:
            return FateResolution(FOUNDER_FATE[name], "founder", 1.0, 0)
        if name in self.leaf_fate:
            return FateResolution(self.leaf_fate[name], "exact", 1.0, 1)

        leaves = self._descendant_leaves(name)
        if leaves:
            counts = Counter(leaves)
            top, n_top = counts.most_common(1)[0]
            purity = n_top / len(leaves)
            fate = top if purity >= MIXED_PURITY else "mixed"
            return FateResolution(fate, "consensus", purity, len(leaves))

        # No descendants (sublineage died out): back off to nearest named ancestor.
        parent = _lineage_parent(name)
        while parent is not None:
            if parent in FOUNDER_FATE and parent not in self.leaf_fate:
                return FateResolution(FOUNDER_FATE[parent], "ancestor", 1.0, 0)
            pleaves = self._descendant_leaves(parent)
            if pleaves:
                counts = Counter(pleaves)
                top, n_top = counts.most_common(1)[0]
                purity = n_top / len(pleaves)
                fate = top if purity >= MIXED_PURITY else "mixed"
                return FateResolution(fate, "ancestor", purity, len(pleaves))
            parent = _lineage_parent(parent)
        return FateResolution("unknown", "unknown", 0.0, 0)


# Local copy of the founder/suffix lineage rule (kept in sync with epic_preprocess).
_FOUNDER_PARENT: dict[str, str] = {
    "AB": "P0", "P1": "P0", "EMS": "P1", "P2": "P1", "MS": "EMS", "E": "EMS",
    "C": "P2", "P3": "P2", "D": "P3", "P4": "P3", "Z2": "P4", "Z3": "P4",
}


def _lineage_parent(cell: str) -> str | None:
    if cell in _FOUNDER_PARENT:
        return _FOUNDER_PARENT[cell]
    if cell and cell[-1].isalpha():
        return cell[:-1]
    return None


def build_fate_index(classes: tuple[str, ...] = FATE_CLASSES) -> dict[str, int]:
    """Stable fate-name -> integer-id map for model targets."""
    return {name: i for i, name in enumerate(classes)}


DEFAULT_FATE_CSV = REPO_ROOT / "Dataset" / "fate" / "cell_fate.csv"

# Fate ids that are not usable supervision targets (ambiguous / no-signal).
NON_TARGET_FATES: frozenset[str] = frozenset({"mixed", "unknown"})


def load_fate_csv(path: str | Path = DEFAULT_FATE_CSV) -> dict[str, str]:
    """Load the generated cell -> fate map (cell name -> fate class name)."""
    df = pd.read_csv(path)
    return dict(zip(df["cell"].astype(str), df["fate"].astype(str)))


def fate_targets(
    idx_to_cell,
    cell_fate: dict[str, str],
    fate_index: dict[str, int] | None = None,
    *,
    exclude: frozenset[str] = NON_TARGET_FATES,
):
    """
    Build per-node fate targets aligned to an embryo's node ordering.

    Args:
        idx_to_cell: sequence of cell-name strings, node index -> name.
        cell_fate: name -> fate-class-name map (see :func:`load_fate_csv`).
        fate_index: fate-name -> id map (defaults to :func:`build_fate_index`).
        exclude: fate names treated as non-targets (masked out of the loss).

    Returns:
        labels: numpy int64 (N,) fate ids (-1 where no usable label).
        valid:  numpy bool  (N,) True where the node has a usable target.
    """
    import numpy as np

    fate_index = fate_index or build_fate_index()
    n = len(idx_to_cell)
    labels = np.full(n, -1, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    for i, cell in enumerate(idx_to_cell):
        fate = cell_fate.get(str(cell))
        if fate is None or fate in exclude:
            continue
        labels[i] = fate_index[fate]
        valid[i] = True
    return labels, valid
