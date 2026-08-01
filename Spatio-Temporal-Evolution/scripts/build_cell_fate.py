"""
Generate the node-level cell-fate label map for all cells observed across the
processed EPIC embryos, using src/cell_fate.CellFateMap.

Outputs:
  Dataset/fate/cell_fate.csv   columns: cell,fate,fate_id,method,purity,n_leaves,n_embryos
and prints a coverage + class-distribution report.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cell_fate import CellFateMap, build_fate_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=Path,
                    default=REPO_ROOT / "Dataset" / "processed" / "by_embryo")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Dataset" / "fate" / "cell_fate.csv")
    args = ap.parse_args()

    fate_map = CellFateMap.from_reference()
    fate_index = build_fate_index()

    npzs = sorted(glob.glob(str(args.processed_dir / "*.npz")))
    if not npzs:
        raise SystemExit(f"No .npz embryos in {args.processed_dir}")

    cell_embryos: Counter[str] = Counter()
    for fp in npzs:
        z = np.load(fp, allow_pickle=True)
        for c in {str(c) for c in z["idx_to_cell"]}:
            cell_embryos[c] += 1

    rows = []
    for cell in sorted(cell_embryos):
        r = fate_map.resolve(cell)
        rows.append({
            "cell": cell,
            "fate": r.fate,
            "fate_id": fate_index[r.fate],
            "method": r.method,
            "purity": round(r.purity, 3),
            "n_leaves": r.n_leaves,
            "n_embryos": cell_embryos[cell],
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    total = len(df)
    print("=" * 68)
    print(f"Cell-fate map: {total} unique cells across {len(npzs)} embryos -> {args.out}")
    print("=" * 68)
    print("\nFate distribution (unique cells):")
    for fate, n in df["fate"].value_counts().items():
        print(f"  {fate:12} {n:5}  ({n/total:5.1%})")
    print("\nResolution method:")
    for m, n in df["method"].value_counts().items():
        print(f"  {m:10} {n:5}  ({n/total:5.1%})")
    unknown = df[df["fate"] == "unknown"]["cell"].tolist()
    print(f"\nUnknown ({len(unknown)}): {unknown[:25]}")
    # Weighted by how often a cell actually appears in embryos:
    appear = (df["fate"].repeat(df["n_embryos"])).value_counts(normalize=True)
    print("\nFate distribution (weighted by embryo occurrences):")
    for fate, frac in appear.items():
        print(f"  {fate:12} {frac:5.1%}")


if __name__ == "__main__":
    main()
