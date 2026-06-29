"""
08_correlation_analysis.py -- Stage 1: Exact Cross-Dataset Correlation Analysis
===============================================================================
Computes exact feature correlation matrices across the complete datasets without
sampling using a single-pass streaming covariance accumulator.

Generates high-resolution correlation heatmaps for:
- CICIDS2017
- CSE-CIC-IDS2018
- Lycos-Unicas-IDS2018

Saves:
- figures/correlation_heatmaps.png
- tables/correlation_matrix_*.csv
"""

import os
import glob
import gc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ─── Setup Paths & Logging ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.dirname(SCRIPT_DIR)
TABLES_DIR = os.path.join(STAGE1_DIR, "tables")
FIGURES_DIR = os.path.join(STAGE1_DIR, "figures")
LOGS_DIR   = os.path.join(STAGE1_DIR, "logs")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "08_correlation_analysis.log"),
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
    return [c for c in num_cols if c.strip() not in ["Label", "label", " Label", "Timestamp", "timestamp"]]

def compute_exact_correlation_matrix(name, path):
    logging.info(f"Computing exact correlation matrix for {name} at {path}")
    print(f"\nComputing streaming correlation matrix for {name} ...")

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

    numeric_cols = get_numeric_columns(path)
    D = len(numeric_cols)
    print(f"  Found {D} numerical features. Initializing streaming accumulators ...")

    # Streaming covariance accumulators
    N = 0
    sum_x = np.zeros(D, dtype=np.float64)
    sum_xxT = np.zeros((D, D), dtype=np.float64)

    for f in files:
        print(f"    Reading chunks from {os.path.basename(f)} ...")
        try:
            for chunk in pd.read_csv(f, usecols=lambda c: c.strip() in numeric_cols, chunksize=500000, low_memory=False, encoding="latin1"):
                chunk.columns = chunk.columns.str.strip()
                # Ensure correct column order
                chunk = chunk.reindex(columns=numeric_cols, fill_value=0.0)
                # Convert to numpy float64 matrix, replacing inf/NaN with 0.0
                X = chunk.values.astype(np.float64)
                X[np.isnan(X)] = 0.0
                X[np.isinf(X)] = 0.0
                
                n_chunk = X.shape[0]
                N += n_chunk
                sum_x += X.sum(axis=0)
                sum_xxT += np.dot(X.T, X)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
            continue

    if N <= 1:
        logging.warning(f"Insufficient samples for {name}")
        return None, numeric_cols

    # Calculate exact mean, covariance, and correlation matrices
    mean_x = sum_x / N
    # Covariance C = (sum_xxT - N * mean_x * mean_x^T) / (N - 1)
    cov = (sum_xxT - N * np.outer(mean_x, mean_x)) / (N - 1)
    
    # Correlation R_ij = C_ij / sqrt(C_ii * C_jj)
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-12 # Prevent division by zero for constant features
    
    corr = cov / np.outer(std, std)
    # Clip to [-1, 1] to handle numerical imprecision
    corr = np.clip(corr, -1.0, 1.0)
    
    df_corr = pd.DataFrame(corr, index=numeric_cols, columns=numeric_cols)
    out_csv = os.path.join(TABLES_DIR, f"correlation_matrix_{name.replace(' ', '_').replace('/', '_')}.csv")
    df_corr.to_csv(out_csv)
    print(f"  Exact correlation matrix saved to {out_csv}")
    
    logging.info(f"Completed exact correlation matrix for {name}")
    return df_corr, numeric_cols

def main():
    print("="*60)
    print(" STAGE 1: EXACT CROSS-DATASET CORRELATION ANALYSIS ")
    print("="*60)
    
    corr_matrices = {}
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist -> {path}")
            continue
        df_corr, cols = compute_exact_correlation_matrix(name, path)
        if df_corr is not None:
            corr_matrices[name] = df_corr

    if corr_matrices:
        print("\nGenerating high-resolution correlation heatmaps ...")
        n_datasets = len(corr_matrices)
        fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 7))
        if n_datasets == 1: axes = [axes]
        
        palettes = ["Blues", "Greens", "Oranges"]

        for i, (name, df_corr) in enumerate(corr_matrices.items()):
            ax = axes[i]
            # Plot heatmap without annotations for clean visualization of 70x70 matrix
            sns.heatmap(df_corr, ax=ax, cmap=palettes[i % len(palettes)], cbar=True,
                        xticklabels=False, yticklabels=False, vmin=-1, vmax=1)
            ax.set_title(f"{name} Correlation Matrix ({df_corr.shape[0]}x{df_corr.shape[1]})", fontsize=15, fontweight="bold", pad=15)
            ax.set_xlabel("Features", fontsize=12)
            if i == 0:
                ax.set_ylabel("Features", fontsize=12)

        plt.suptitle("Stage 1: Complete Cross-Dataset Feature Correlation Heatmaps", fontsize=22, fontweight="bold", y=1.05)
        plt.tight_layout()
        
        out_img = os.path.join(FIGURES_DIR, "correlation_heatmaps.png")
        plt.savefig(out_img, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Visualizations saved to {out_img}")

    print("\nDone!")

if __name__ == "__main__":
    main()
