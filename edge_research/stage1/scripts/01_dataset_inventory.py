"""
01_dataset_inventory.py -- Stage 1: Dataset Inventory Audit
===========================================================
Generates foundational dataset statistics across full datasets without sampling:
- Number of rows
- Number of columns
- Dataset size on disk
- Memory usage estimation
- Class count
- Feature count

Saves results to tables/dataset_inventory.csv
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
    filename=os.path.join(LOGS_DIR, "01_dataset_inventory.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

def format_bytes(size):
    # Formats bytes into a human-readable string
    power = 2**10
    n = 0
    power_labels = {0 : 'Bytes', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power and n < 4:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def audit_dataset(name, path):
    logging.info(f"Starting audit for {name} at {path}")
    print(f"Auditing {name} ...")

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

    total_size_bytes = sum(os.path.getsize(f) for f in files)
    total_rows = 0
    unique_classes = set()
    num_columns = 0
    estimated_memory = 0

    for f in files:
        print(f"  Reading {os.path.basename(f)} ...")
        # Peek at header to determine columns and label column name
        peek = pd.read_csv(f, nrows=1, encoding="latin1")
        peek.columns = peek.columns.str.strip()
        num_columns = len(peek.columns)
        
        label_col = None
        for candidate in ["Label", "label", " Label"]:
            if candidate in peek.columns:
                label_col = candidate
                break

        # If no explicit label column found, take the last column as label
        if label_col is None:
            label_col = peek.columns[-1]

        # Read only the label column in chunks to save memory while getting exact row/class counts
        for chunk in pd.read_csv(f, usecols=lambda c: c.strip() == label_col, chunksize=500000, low_memory=False, encoding="latin1"):
            chunk.columns = chunk.columns.str.strip()
            total_rows += len(chunk)
            # Accumulate unique classes, filtering out potential header repetitions in raw CSVs
            val_counts = chunk[label_col].astype(str).str.strip().unique()
            unique_classes.update([val for val in val_counts if val not in ["Label", "label"]])

    # Approximate memory usage of full dataframe in memory (float64/int64 assumption: ~8 bytes per cell + overhead)
    estimated_memory_bytes = total_rows * num_columns * 8
    
    result = {
        "Dataset": name,
        "Number of Rows": total_rows,
        "Number of Columns": num_columns,
        "Feature Count": num_columns - 1 if num_columns > 0 else 0,
        "Class Count": len(unique_classes),
        "Dataset Size (Disk)": format_bytes(total_size_bytes),
        "Estimated Memory Usage (RAM)": format_bytes(estimated_memory_bytes),
        "Unique Classes": "; ".join(sorted([str(x) for x in unique_classes if str(x) != 'nan']))
    }
    
    logging.info(f"Completed audit for {name}: {result}")
    return result

def main():
    print("="*60)
    print(" STAGE 1: DATASET INVENTORY AUDIT ")
    print("="*60)
    
    results = []
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist -> {path}")
            logging.error(f"Path not found: {path}")
            continue
        res = audit_dataset(name, path)
        results.append(res)
        
    df_inv = pd.DataFrame(results)
    
    out_csv = os.path.join(TABLES_DIR, "dataset_inventory.csv")
    df_inv.to_csv(out_csv, index=False)
    print(f"\nInventory saved to {out_csv}")
    print("\nSummary Table:")
    print(df_inv[["Dataset", "Number of Rows", "Number of Columns", "Feature Count", "Class Count", "Dataset Size (Disk)"]].to_string(index=False))
    print("\nDone!")

if __name__ == "__main__":
    main()
