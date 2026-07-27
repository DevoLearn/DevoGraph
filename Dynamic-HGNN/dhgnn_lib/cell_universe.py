"""Fixed cell-name universe for the DHGNN embryogenesis pipeline.

Builds a single, stable mapping from cell name -> integer index that every
other module (node features, hyperedge builders, incidence assembly) relies
on so that node ordering is consistent across the whole pipeline.

The universe is the sorted union of every cell name that appears anywhere in
the raw dataset or the two pre-existing lookup tables:
  - ce_temporal_data.csv          (cell, mother)
  - cells_birth_and_pos.csv       (Parent Cell, Daughter 1, Daughter 2)
  - CE_cell_graph_data_processed.csv (cell, mother)
  - functional_lookup_table.csv   (embryo_A/B/C_sulston)
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

# Founder lineage names in C. elegans Sulston nomenclature. Sorted longest
# first so prefix matching picks "EMS" before "E", etc.
_FOUNDERS = sorted(
    ["EMS", "AB", "MS", "E", "C", "D", "Z1", "Z2", "Z3", "P0", "P1", "P2", "P3", "P4"],
    key=len,
    reverse=True,
)

# Characters that denote one round of cell division in Sulston names
# (anterior/posterior/left/right/dorsal/ventral).
_DIVISION_CHARS = set("aplrdv")


def lineage_generation_from_name(name: str) -> int:
    """Return the lineage depth (number of cell divisions since the founder).

    For a Sulston-style name such as ``ABplapaaaapp`` this strips the founder
    prefix (``AB``) and counts the remaining division characters
    (``plapaaaapp`` -> 10). Founder cells themselves (``AB``, ``P1``, ...)
    have generation 0. Names that do not match a known founder prefix fall
    back to the length of the longest trailing run of division characters.
    """
    if not isinstance(name, str) or not name:
        return 0

    for founder in _FOUNDERS:
        if name.startswith(founder):
            suffix = name[len(founder):]
            if suffix == "" or all(c in _DIVISION_CHARS for c in suffix):
                return len(suffix)

    depth = 0
    for c in reversed(name):
        if c in _DIVISION_CHARS:
            depth += 1
        else:
            break
    return depth


def _split_members(cell_str) -> List[str]:
    if not isinstance(cell_str, str) or not cell_str:
        return []
    return [c for c in cell_str.split("|") if c]


def build_cell_universe(raw_dir: str, output_dir: str) -> List[str]:
    """Return the sorted list of every cell name in the project's universe."""
    names: set = set()

    ctd = pd.read_csv(os.path.join(raw_dir, "ce_temporal_data.csv"))
    names.update(ctd["cell"].dropna().astype(str).unique())
    names.update(ctd["mother"].dropna().astype(str).unique())

    cbp = pd.read_csv(os.path.join(raw_dir, "cells_birth_and_pos.csv"))
    names.update(cbp["Parent Cell"].dropna().astype(str).unique())
    names.update(cbp["Daughter 1"].dropna().astype(str).unique())
    names.update(cbp["Daughter 2"].dropna().astype(str).unique())

    cgd = pd.read_csv(os.path.join(raw_dir, "CE_cell_graph_data_processed.csv"))
    names.update(cgd["cell"].dropna().astype(str).unique())
    names.update(cgd["mother"].dropna().astype(str).unique())

    func = pd.read_csv(os.path.join(output_dir, "functional_lookup_table.csv"))
    for col in ("embryo_A_sulston", "embryo_B_sulston", "embryo_C_sulston"):
        names.update(func[col].dropna().astype(str).unique())

    names.discard("")
    names = {n for n in names if n.lower() != "nan"}

    return sorted(names)


class CellUniverse:
    """Stable cell-name <-> index mapping shared across the pipeline."""

    def __init__(self, names: List[str]):
        self.names: List[str] = list(names)
        self.index: Dict[str, int] = {name: i for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str) -> int:
        return self.index[name]

    @classmethod
    def build(cls, raw_dir: str, output_dir: str) -> "CellUniverse":
        return cls(build_cell_universe(raw_dir, output_dir))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "index": list(range(len(self.names))),
                "cell": self.names,
                "lineage_generation": [lineage_generation_from_name(n) for n in self.names],
            }
        )
