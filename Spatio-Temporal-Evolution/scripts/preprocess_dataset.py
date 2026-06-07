from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.epic_preprocess import preprocess_epic_file_sparse 


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("dataset/raw"),
        help="Directory containing EPIC *.csv files",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/processed/by_embryo"),
        help="Output directory (one .npz per embryo)",
    )
    ap.add_argument("--distance_threshold", type=float, default=20.0)
    args = ap.parse_args()

    files = sorted(args.raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No .csv files found in {args.raw_dir}")

    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for fp in tqdm(files, desc="Preprocessing EPIC files"):
        X, alive, edge_src, edge_dst, edge_t, index = preprocess_epic_file_sparse(
            fp,
            distance_threshold=args.distance_threshold,
        )
        idx_to_cell = np.array(
            [c for c, _ in sorted(index.cell_to_idx.items(), key=lambda kv: kv[1])],
            dtype=object,
        )
        out_fp = args.out / f"{fp.stem}.npz"
        np.savez_compressed(
            out_fp,
            X=X,
            alive_mask=alive,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_t=edge_t,
            idx_to_cell=idx_to_cell,
            t0=np.int32(index.t0),
            T=np.int32(index.T),
            source_file=fp.name,
        )
        manifest.append(out_fp.name)

    (args.out / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

