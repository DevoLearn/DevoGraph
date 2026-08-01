from __future__ import annotations

import csv
from pathlib import Path

# Heuristic wild-type / control keywords in EPIC filenames (GSoC proposal: WT vs RNAi).
_CONTROL_KEYWORDS: tuple[str, ...] = (
    "bright",
    "norfp",
    "norfpxx",
    "norfpyy",
    "5a_bright",
    "end1red",
    "end3",
    "rw10029",
)


def infer_label_from_filename(filename: str) -> int:
    """
    Binary label: 0 = control / wild-type-like, 1 = perturbed / RNAi-like.

    Override with a CSV manifest via ``load_label_map`` when heuristics are insufficient.
    """
    low = Path(filename).stem.lower()
    if any(keyword in low for keyword in _CONTROL_KEYWORDS):
        return 0
    return 1


def load_label_map(
    processed_dir: str | Path,
    manifest_path: str | Path | None = None,
    labels_csv: str | Path | None = None,
) -> dict[str, int]:
    """
    Build filename -> label mapping.

    If ``labels_csv`` is provided it must contain columns ``filename,label``.
    Otherwise heuristic labels are inferred from filenames.
    """
    processed_dir = Path(processed_dir)
    if manifest_path is None:
        manifest_path = processed_dir / "manifest.txt"
    manifest_path = Path(manifest_path)

    filenames = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    label_map = {name: infer_label_from_filename(name) for name in filenames}

    if labels_csv is not None:
        labels_csv = Path(labels_csv)
        with labels_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"].strip()
                label_map[fname] = int(row["label"])

    return label_map
