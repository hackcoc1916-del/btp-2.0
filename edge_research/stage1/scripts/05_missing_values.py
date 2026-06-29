"""
05_missing_values.py -- Stage 1: Missing & Invalid Values Audit
===============================================================
Calculates exact NaN count, inf count, and -inf count for every feature across
the complete datasets without sampling.

Generates table with columns:
- Dataset
- Feature
- NaN Count
- Inf Count
- -Inf Count
- Total Invalid

Saves:
- tables/missing_values.csv
"""

import os
import glob
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
    filename=os.path.join(LOGS_DIR, "05_missing_values.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

def audit_missing_values(name, path):
    logging.info(f"Auditing missing values for {name} at {path}")
    print(f"Scanning missing/invalid values for {name} ...")

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

    nan_counts = pd.Series(dtype=int)
    inf_counts = pd.Series(dtype=int)
    ninf_counts = pd.Series(dtype=int)

    for f in files:
        print(f"  Reading {os.path.basename(f)} ...")
        for chunk in pd.read_csv(f, chunksize=500000, low_memory=False, encoding="latin1"):
            chunk.columns = chunk.columns.str.strip()
            
            # Identify numeric columns for inf/-inf checking
            # Non-numeric columns won't have np.inf
            # But we check isna() on all columns
            nans = chunk.isna().sum()
            nan_counts = nan_counts.add(nans, fill_value=0)

            # Check inf/-inf on numeric columns
            # Select only numeric types or convert to numeric where possible
            num_chunk = chunk.select_dtypes(include=[np.number])
            if not num_chunk.empty:
                infs = (num_chunk == np.inf).sum()
                ninfs = (num_chunk == -np.inf).sum()
                inf_counts = inf_counts.add(infs, fill_value=0)
                ninf_counts = ninf_counts.add(ninfs, fill_value=0)

    df_res = pd.DataFrame({
        "Dataset": name,
        "Feature": nan_counts.index,
        "NaN Count": nan_counts.values.astype(int),
        "Inf Count": inf_counts.reindex(nan_counts.index, fill_value=0).values.astype(int),
        "-Inf Count": ninf_counts.reindex(nan_counts.index, fill_value=0).values.astype(int)
    })
    
    df_res["Total Invalid"] = df_res["NaN Count"] + df_res["Inf Count"] + df_res["-Inf Count"]
    df_res = df_res.sort_values(by="Total Invalid", ascending=False)

    logging.info(f"Completed missing values audit for {name}")
    return df_res

def main():
    print("="*60)
    print(" STAGE 1: MISSING & INVALID VALUES AUDIT ")
    print("="*60)
    
    all_results = []
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist -> {path}")
            continue
        df_res = audit_missing_values(name, path)
        all_results.append(df_res)
        
    if all_results:
        df_full = pd.concat(all_results, ignore_index=True)
        out_csv = os.path.join(TABLES_DIR, "missing_values.csv")
        df_full.to_csv(out_csv, index=False)
        print(f"\nMissing values report saved to {out_csv}")
        
        # Print summary of features with invalid values
        df_invalid = df_full[df_full["Total Invalid"] > 0]
        print("\nFeatures Containing Invalid Values (NaN, inf, -inf):")
        if not df_invalid.empty:
            print(df_invalid.to_string(index=False))
        else:
            print("  SUCCESS: No invalid values found across any datasets!")

    print("\nDone!")

if __name__ == "__main__":
    main()
