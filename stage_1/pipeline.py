"""
Stage 1 Pipeline Orchestrator

Wraps segment.py -> extract.py -> convert.py into a single end-to-end
command that takes raw microscopy images and outputs a Stage 2-ready CSV.

Segmentation masks are saved to a user-specified directory so results
can be visually inspected.

Usage:
    python stage_1/pipeline.py --input data/Fluo-N2DH-SIM+/01 \
                               --masks-dir output/masks \
                               --output output.csv
"""

import os
import glob
import argparse
import re

from stage_1.segment import WatershedSegmenter
from stage_1.extract import extract_features
from stage_1.convert import to_devograph_schema

import pandas as pd


def run_pipeline(input_dir, masks_dir, output_csv,
                 threshold=10, image_ext="tif"):
    """
    Full Stage 1 pipeline:
      1. Segment each frame using devolearn + watershed
      2. Extract features (centroids, area) from each mask
      3. Convert to devograph schema (cell, time, x, y, z, size)

    Masks are saved to masks_dir for visual inspection.

    Args:
        input_dir:  Directory containing raw microscopy images
        masks_dir:  Directory to save segmentation mask images
        output_csv: Path for the consolidated output CSV
        threshold:  Binary threshold value for watershed
        image_ext:  Image file extension to look for (default: "tif")
    """
    patterns = [
        os.path.join(input_dir, f"*.{image_ext}"),
        os.path.join(input_dir, f"*.{image_ext.upper()}"),
    ]
    # check formats
    for ext in ["png", "tiff"]:
        if ext != image_ext.lower():
            patterns.append(os.path.join(input_dir, f"*.{ext}"))

    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(pattern))

    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

    image_files = sorted(set(image_files), key=natural_keys)

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")

    # Create output directories
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # Create segmenter
    segmenter = WatershedSegmenter(threshold=threshold)

    # Process each frame
    all_features = []
    for frame_num, image_path in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(masks_dir, f"{basename}_mask.png")

        print(f"\n--- Frame {frame_num}: {os.path.basename(image_path)} ---")

        # Step 1: Segment
        label_mask = segmenter.segment(image_path, mask_path)
        if label_mask is None:
            print(f"  Skipping frame {frame_num} (segmentation failed)")
            continue

        num_cells = len(set(label_mask.flat)) - 1  # exclude background
        print(f"  Segmented: {num_cells} cells found")

        # Step 2: Extract features
        df = extract_features(mask_path, frame_num=frame_num)
        if df is not None and not df.empty:
            all_features.append(df)
            print(f"  Extracted: {len(df)} cell features")
        else:
            print(f"  No features extracted from frame {frame_num}")

    if not all_features:
        print("\nNo features extracted from any frame.")
        return

    # Combine all per-frame features
    combined_df = pd.concat(all_features, ignore_index=True)

    # Step 3: Convert to devograph schema
    mapped_df = to_devograph_schema(combined_df)

    # Write output
    mapped_df.to_csv(output_csv, index=False)

    print(f"\n{'='*50}")
    print(f"Pipeline complete!")
    print(f"  Total frames processed: {len(all_features)}")
    print(f"  Total cells across all frames: {len(mapped_df)}")
    print(f"  Output columns: {list(mapped_df.columns)}")
    print(f"  Masks saved to: {masks_dir}")
    print(f"  CSV saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 Pipeline: raw images -> devograph-ready CSV"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory containing raw microscopy images"
    )
    parser.add_argument(
        "--masks-dir", "-d", type=str, required=True,
        help="Directory to save segmentation mask images"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Path for output CSV (devograph schema)"
    )
    parser.add_argument(
        "--threshold", "-t", type=int, default=10,
        help="Binary threshold for watershed (default: 10)"
    )
    parser.add_argument(
        "--ext", type=str, default="tif",
        help="Image file extension to look for (default: tif)"
    )

    args = parser.parse_args()
    run_pipeline(args.input, args.masks_dir, args.output,
                 threshold=args.threshold, image_ext=args.ext)
