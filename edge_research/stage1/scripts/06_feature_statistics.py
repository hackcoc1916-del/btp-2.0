"""
06_feature_statistics.py -- Stage 1: Full Dataset Numerical Statistics Audit
============================================================================
Calculates exact statistical properties for every numerical feature across the
complete datasets without sampling:
- mean
- std
- median
- min
- max
- skewness
- kurtosis

Utilizes single-column streaming (`usecols`) to process 15M+ rows efficiently
without exceeding memory limits.

Saves:
- tables/feature_statistics.csv
"""

import os
import glob
import gc
import pandas as pd
import numpy as np
import logging

# ─── Setup Paths & Logging ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.dirname(SCRIPT_DIR)
TABLES_DIR = os.path.join(STAGE1_DIR, "tables")
LOGS_DIR   = os.path.join(STAGE1_DIR, "logs")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "06_feature_statistics.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

def get_numeric_columns(path):
    if os.path.isfile(path):
        f = path
    else:
        f = sorted(glob.glob(os.path.join(path, "*.csv")))[0]
    peek = pd.read_csv(f, nrows=100, encoding="latin1")
    peek.columns = peek.columns.str.strip()
    num_cols = peek.select_dtypes(include=[np.number]).columns.tolist()
    # Remove label/timestamp if present in numeric form
    return [c for c in num_cols if c.strip() not in ["Label", "label", " Label", "Timestamp", "timestamp"]]

def calculate_feature_stats(name, path):
    logging.info(f"Starting feature statistics calculation for {name} at {path}")
    print(f"\nCalculating numerical feature statistics for {name} ...")

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

    numeric_cols = get_numeric_columns(path)
    print(f"  Found {len(numeric_cols)} numerical columns to analyze.")

    dataset_results = []

    # Process one column at a time to keep RAM usage under 200MB even for 15M rows
    for col in numeric_cols:
        print(f"    Processing feature: {col} ...")
        series_chunks = []
        for f in files:
            try:
                # Read only the target column
                for chunk in pd.read_csv(f, usecols=lambda c: c.strip() == col, chunksize=1000000, low_memory=False, encoding="latin1"):
                    chunk.columns = chunk.columns.str.strip()
                    s = pd.to_numeric(chunk[col], errors="coerce")
                    # Drop NaN/inf for statistical accuracy
                    s = s.replace([np.inf, -np.inf], np.nan).dropna()
                    series_chunks.append(s)
            except Exception as e:
                logging.error(f"Error reading {col} in {f}: {e}")
                continue

        if not series_chunks:
            continue

        full_series = pd.concat(series_chunks, ignore_index=True)
        
        if full_series.empty:
            continue

        stats = {
            "Dataset": name,
            "Feature": col,
            "mean": full_series.mean(),
            "std": full_series.std(),
            "median": full_series.median(),
            "min": full_series.min(),
            "max": full_series.max(),
            "skewness": full_series.skew(),
            "kurtosis": full_series.kurt()
        }
        dataset_results.append(stats)

        # Explicit memory cleanup
        del full_series, series_chunks
        gc.collect()

    df_stats = pd.DataFrame(dataset_results)
    logging.info(f"Completed statistics calculation for {name}")
    return df_stats

def main():
    print("="*60)
    print(" STAGE 1: FULL DATASET NUMERICAL STATISTICS AUDIT ")
    print("="*60)
    
    all_dfs = []
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist -> {path}")
            continue
        df_stats = calculate_feature_stats(name, path)
        all_dfs.append(df_stats)
        
    if all_dfs:
        df_full = pd.concat(all_dfs, ignore_index=True)
        out_csv = os.path.join(TABLES_DIR, "feature_statistics.csv")
        df_full.to_csv(out_csv, index=False)
        print(f"\nFeature statistics successfully saved to {out_csv}")
        print("\nSample Statistics (First 10 Features):")
        print(df_full.head(10).to_string(index=False))

    print("\nDone!")

if __name__ == "__main__":
    main()
