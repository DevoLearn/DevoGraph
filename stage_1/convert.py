import pandas as pd
import os
import glob

def to_devograph_schema(stage1_df):
    """
    Maps Stage 1 tracking columns to Stage 2 (`devograph/datasets/datasets.py`) expectations.
    """
    df = pd.DataFrame()
    df['cell'] = stage1_df['id']
    df['time'] = stage1_df['frame_num']
    # Mapping coordinates depending on 2D extraction
    df['x'] = stage1_df['centroid_row'] 
    df['y'] = stage1_df['centroid_col']
    df['z'] = 0  # Default for 2D data
    df['size'] = stage1_df['area']
    return df

def convert_directory(input_dir, output_csv_path):
    """
    Reads all per-frame CSVs from process_seq_2d and outputs a single combined CSV.
    """
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return
        
    dfs = []
    for f in all_files:
        dfs.append(pd.read_csv(f))
        
    combined_df = pd.concat(dfs, ignore_index=True)
    mapped_df = to_devograph_schema(combined_df)
    
    mapped_df.to_csv(output_csv_path, index=False)
    print(f"Successfully converted {len(all_files)} files to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Stage 1 CSVs to Stage 2 schema")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing per-frame CSVs")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path for output consolidated CSV")
    
    args = parser.parse_args()
    convert_directory(args.input, args.output)
