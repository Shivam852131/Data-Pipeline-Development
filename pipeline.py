
#!/usr/bin/env python3
"""
pipeline.py

Simple ETL pipeline for "Data Pipeline Development" (CODTECH Internship Task-1).
- Extracts data from a CSV file
- Transforms the data (missing values, label encoding, scaling)
- Loads the processed data to a CSV file

Usage:
    python pipeline.py --input data/students.csv --output data/processed_students.csv

This script is intentionally simple and written in a clear, modular style suitable
for a B.Tech internship submission.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------
# Helper functions
# ---------------------------
def extract_data(file_path: str) -> pd.DataFrame:
    """Read CSV into a pandas DataFrame."""
    print(f"[ETL] Extracting data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[ETL] Extracted {len(df)} rows and {len(df.columns)} columns.")
    return df


def transform_data(df: pd.DataFrame):
    """
    Clean, encode, and scale the DataFrame.
    Returns the transformed DataFrame and a small metadata dict containing encoders/scaler info.
    """
    print("[ETL] Transforming data...")

    # Make a copy to avoid modifying original
    df = df.copy()

    # 1) Handle missing values
    # For numeric columns use mean, for categorical use most frequent
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    else:
        num_imputer = None

    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    else:
        cat_imputer = None

    # 2) Encode categorical columns using LabelEncoder (simple and reproducible)
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 3) Scale numeric columns using StandardScaler
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        scaler = None

    metadata = {
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "num_imputer": "mean" if num_imputer is not None else None,
        "cat_imputer": "most_frequent" if cat_imputer is not None else None,
        "scaler": "StandardScaler" if scaler is not None else None,
        "label_encoders": list(label_encoders.keys()),
    }

    print("[ETL] Transformation complete.")
    return df, metadata, label_encoders, scaler


def load_data(df: pd.DataFrame, output_path: str):
    """Save the DataFrame to CSV."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[ETL] Loaded processed data to: {output_path}")


def save_metadata(metadata: dict, output_path: str):
    """Save simple metadata (columns, encoders used) to a JSON file alongside output CSV."""
    meta_path = Path(output_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[ETL] Saved metadata to: {meta_path}")


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Simple ETL pipeline for internship task.")
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to output CSV file')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = extract_data(input_path)
    df_processed, metadata, label_encoders, scaler = transform_data(df)
    load_data(df_processed, output_path)
    save_metadata(metadata, output_path)

    print("[ETL] Pipeline finished successfully.")


if __name__ == '__main__':
    main()
