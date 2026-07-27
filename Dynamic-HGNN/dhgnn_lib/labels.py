"""Per-cell label tensor used by the DHGNN.

  - ``fate``: static per-cell founder-lineage class (AB/MS/E/C/D/P*/Z*),
    the supervised cell-fate classification target (L_fate, cross-entropy).
"""

from __future__ import annotations

from typing import List

import torch

from .cell_universe import CellUniverse, _FOUNDERS

FATE_NAMES: List[str] = list(_FOUNDERS) + ["unknown"]
_FATE_INDEX = {name: i for i, name in enumerate(FATE_NAMES)}
_UNKNOWN_FATE = _FATE_INDEX["unknown"]


def _founder_of(name: str) -> str:
    if not isinstance(name, str) or not name:
        return "unknown"
    for founder in _FOUNDERS:
        if name.startswith(founder):
            return founder
    return "unknown"


def fate_labels(universe: CellUniverse) -> torch.Tensor:
    """Return (N,) LongTensor of founder-lineage class indices, see ``FATE_NAMES``."""
    idx = [_FATE_INDEX.get(_founder_of(name), _UNKNOWN_FATE) for name in universe.names]
    return torch.tensor(idx, dtype=torch.long)
