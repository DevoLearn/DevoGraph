"""
Assemble a self-contained folder to upload to a Hugging Face Space.

The API needs the source modules, the trained checkpoint and the fate label map
sitting next to app.py. Those live in different places in this repo (and *.pt is
git-ignored), so this collects them into api/ ready to push to the Space.

    python scripts/build_api_bundle.py

Then push api/ to your Space repo (see api/README.md).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
API = REPO / "api"
SPACE = REPO / "space"

SRC_MODULES = [
    "__init__.py", "cell_fate.py", "epic_preprocess.py",
    "estgel_eam.py", "estgel_layers.py", "estgel_node.py",
]
MODEL_FILES = [
    (REPO / "checkpoints" / "estgel_fate" / "best.pt", "best.pt"),
    (REPO / "checkpoints" / "estgel_fate" / "config.json", "config.json"),
    (REPO / "checkpoints" / "estgel_fate" / "history.json", "history.json"),
    (REPO / "checkpoints" / "estgel_fate" / "per_class_recall.json", "per_class_recall.json"),
    (REPO / "Dataset" / "fate" / "cell_fate.csv", "cell_fate.csv"),
]


def main() -> None:
    for target in (API, SPACE):
        _fill(target)
    # the Gradio Space reuses the FastAPI inference module
    shutil.copy2(API / "service.py", SPACE / "service.py")
    _make_example()


def _make_example() -> None:
    """Trim one raw EPIC embryo down to the required columns as a demo input."""
    import pandas as pd

    raw = sorted((REPO / "Dataset" / "raw").glob("*.csv"))
    if not raw:
        print("No Dataset/raw CSVs; skipped example."); return
    out_dir = SPACE / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["cell", "time", "x", "y", "z", "size", "blot"]
    df = pd.read_csv(raw[0])[cols]
    df.to_csv(out_dir / "wildtype_embryo.csv", index=False)
    print(f"  example: {out_dir / 'wildtype_embryo.csv'} ({len(df)} rows)")


def _fill(target: Path) -> None:
    src_out = target / "src"
    model_out = target / "model"
    src_out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in SRC_MODULES:
        s = REPO / "src" / name
        if not s.exists():
            missing.append(str(s)); continue
        shutil.copy2(s, src_out / name)

    total = 0
    for s, dest in MODEL_FILES:
        if not s.exists():
            missing.append(str(s)); continue
        shutil.copy2(s, model_out / dest)
        total += (model_out / dest).stat().st_size

    if missing:
        print("MISSING (train the model first?):")
        for m in missing:
            print("  -", m)
        sys.exit(1)

    print(f"Bundle ready: {target}")
    print(f"  src/    {len(SRC_MODULES)} modules")
    print(f"  model/  {len(MODEL_FILES)} files, {total/1024:.0f} KB")
    print("\nUpload the contents of api/ to your Hugging Face Space (Docker SDK).")


if __name__ == "__main__":
    main()
