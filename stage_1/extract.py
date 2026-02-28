import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label as sk_label
import argparse


def _ensure_instance_labels(mask):
    """
    If the mask is binary (only one non-zero value like 255), apply
    connected-component labeling so each individual cell gets a unique ID.
    Instance-labeled masks (multiple distinct non-zero values) are kept as-is.
    """
    unique_vals = np.unique(mask)
    non_zero = unique_vals[unique_vals != 0]

    if len(non_zero) <= 1:
        # Binary mask — all foreground shares a single value.
        # Use connected components to assign unique IDs per cell.
        binary = (mask > 0).astype(np.uint8)
        num_labels, labeled = cv2.connectedComponents(binary)
        print(f"  Binary mask detected (value={non_zero[0] if len(non_zero) else 'none'}). "
              f"Connected-component labeling found {num_labels - 1} cells.")
        return labeled.astype(np.int32)
    else:
        # Already instance-labeled (each cell has a unique pixel value)
        print(f"  Instance-labeled mask detected with {len(non_zero)} unique cell IDs.")
        return mask


def extract_features(mask_path, frame_num=0):
    """
    Extracts basic features (centroid, area, etc.) from a segmentation mask
    using skimage's regionprops.

    Handles both:
      - Instance-labeled masks (each cell = unique pixel value)
      - Binary masks (all cells = same value, e.g. 255)
    """
    # Read mask preserving bit depth. Many instance-label masks are 16-bit.
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"Error: Could not read mask at {mask_path}")
        return None

    # If mask is RGB/RGBA, convert to single channel.
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Ensure we operate on an integer label image.
    if mask.dtype.kind == "f":
        mask = np.rint(mask).astype(np.int32)

    # Ensure each cell has a unique label
    labeled_mask = _ensure_instance_labels(mask)

    features = []
    for cell_id in np.unique(labeled_mask):
        if cell_id == 0:
            continue  # skip background

        # Isolate this cell — same approach as process_seq_2d
        cell_binary = np.uint8(labeled_mask == cell_id)
        props = regionprops(cell_binary)

        if not props:
            continue

        prop = props[0]

        # Regionprops returns centroid as (row, col)
        centroid_row, centroid_col = prop.centroid
        area = prop.area

        features.append({
            'id': cell_id,
            'frame_num': frame_num,
            'centroid_row': centroid_row,
            'centroid_col': centroid_col,
            'area': area
        })

    df = pd.DataFrame(features)
    return df

def run_extraction(mask_path, output_csv, frame_num=0):
    """
    Runs extraction on a single mask and saves to CSV.
    """
    df = extract_features(mask_path, frame_num)
    if df is not None and not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Successfully extracted {len(df)} features to {output_csv}")
    else:
        print(f"No features found : {mask_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a segmentation mask")
    parser.add_argument("--mask", "-m", type=str, required=True, help="Path to input mask image")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--frame", "-f", type=int, default=0, help="Frame number to embed in CSV")
    
    args = parser.parse_args()
    run_extraction(args.mask, args.output, args.frame)
