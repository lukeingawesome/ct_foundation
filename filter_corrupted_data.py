#!/usr/bin/env python3
"""
Filter corrupted/unreadable NPZ files from dataset
==================================================

This script:
1. Reads the CSV file
2. Checks each NPZ file for readability
3. Removes rows with corrupted/unreadable files
4. Saves a cleaned CSV file

Usage:
    python filter_corrupted_data.py --input_csv path/to/input.csv --output_csv path/to/cleaned.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

def check_npz_file(file_path):
    """
    Check if an NPZ file can be opened and contains the expected 'image' key.
    
    Args:
        file_path (str): Path to the NPZ file
        
    Returns:
        bool: True if file is readable and valid, False otherwise
    """
    try:
        # Check if file exists
        if not Path(file_path).exists():
            return False
            
        # Try to open and read the NPZ file
        with np.load(file_path) as npz:
            # Check if 'image' key exists
            if 'image' not in npz:
                return False
                
            # Try to access the data to ensure it's readable
            arr = npz['image']
            
            # Basic shape validation
            if arr.ndim != 4:  # Expected: (C, D, H, W)
                return False
                
            # Check for reasonable dimensions
            if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0 or arr.shape[3] == 0:
                return False
                
            return True
            
    except (EOFError, OSError, ValueError, KeyError, Exception) as e:
        return False

def filter_corrupted_data(input_csv, output_csv, img_key='img_path'):
    """
    Filter out rows with corrupted or unreadable NPZ files.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output cleaned CSV file
        img_key (str): Column name containing image paths
    """
    # Load the CSV
    print(f"Loading CSV from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Original dataset size: {len(df)} rows")
    
    # Check if required columns exist
    required_cols = [img_key, 'instruction', 'answer', 'split']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
    
    # Filter out rows with corrupted files
    print("Checking NPZ files...")
    valid_rows = []
    corrupted_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        img_path = row[img_key]
        
        if pd.isna(img_path) or not isinstance(img_path, str):
            print(f"Row {idx}: Invalid image path: {img_path}")
            corrupted_count += 1
            continue
            
        if check_npz_file(img_path):
            valid_rows.append(idx)
        else:
            print(f"Row {idx}: Corrupted/unreadable file: {img_path}")
            corrupted_count += 1
    
    # Create cleaned dataframe
    cleaned_df = df.iloc[valid_rows].reset_index(drop=True)
    
    print(f"\nResults:")
    print(f"Original rows: {len(df)}")
    print(f"Corrupted/unreadable files: {corrupted_count}")
    print(f"Valid rows: {len(cleaned_df)}")
    print(f"Removed {corrupted_count} rows ({corrupted_count/len(df)*100:.1f}%)")
    
    # Save cleaned dataframe
    print(f"\nSaving cleaned CSV to: {output_csv}")
    cleaned_df.to_csv(output_csv, index=False)
    print("Done!")
    
    # Print some statistics by split
    if 'split' in cleaned_df.columns:
        print(f"\nSplit distribution in cleaned data:")
        split_counts = cleaned_df['split'].value_counts()
        for split, count in split_counts.items():
            print(f"  {split}: {count} samples")

def main():
    parser = argparse.ArgumentParser(description="Filter corrupted NPZ files from dataset")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to output cleaned CSV file")
    parser.add_argument("--img_key", default="img_path", help="Column name for image paths (default: img_path)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # Run the filtering
    filter_corrupted_data(args.input_csv, args.output_csv, args.img_key)

if __name__ == "__main__":
    main() 