from __future__ import annotations

import json
import shutil
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

def export_embryo(npz_path: Path, sampled_ts: list[int]) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    x = data["X"]  # shape: (N, 5, T)
    alive_mask = data["alive_mask"]  # shape: (N, T)
    idx_to_cell = data["idx_to_cell"]  # shape: (N,)
    T = int(data["T"])
    
    ts_data = {}
    for t in sampled_ts:
        if t >= T:
            continue
            
        active_indices = np.where(alive_mask[:, t])[0]
        cells_at_t = []
        for idx in active_indices:
            cells_at_t.append({
                "name": str(idx_to_cell[idx]),
                "pos": [float(x[idx, 0, t]), float(x[idx, 1, t]), float(x[idx, 2, t])],
                "size": float(x[idx, 3, t]),
                "blot": float(x[idx, 4, t])
            })
        ts_data[str(t)] = cells_at_t
        
    return {
        "filename": npz_path.name,
        "total_timesteps": T,
        "timesteps": ts_data
    }

def main() -> None:
    processed_dir = REPO_ROOT / "Dataset" / "processed" / "by_embryo"
    wt_path = processed_dir / "CD011505_end1red_bright.npz"
    rnai_path = processed_dir / "CD030906_dyf7red.npz"
    
    dashboard_dir = REPO_ROOT / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    
    sampled_ts = [30, 60, 90, 120, 150, 180]
    
    print(f"Exporting WT embryo data from {wt_path.name}...")
    wt_data = export_embryo(wt_path, sampled_ts)
    
    print(f"Exporting RNAi embryo data from {rnai_path.name}...")
    rnai_data = export_embryo(rnai_path, sampled_ts)
    
    output = {
        "wt": wt_data,
        "rnai": rnai_data
    }
    
    output_path = dashboard_dir / "embryo_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved exported embryo details to {output_path}")
    
    # Copy explainability report to dashboard directory
    report_src = REPO_ROOT / "checkpoints" / "estgel" / "explainability_report.json"
    report_dst = dashboard_dir / "explainability_report.json"
    
    if report_src.exists():
        shutil.copy(report_src, report_dst)
        print(f"Copied explainability report to {report_dst}")
    else:
        print(f"Warning: {report_src} does not exist. Did you run explainability extraction?")

if __name__ == "__main__":
    main()
