"""
02_class_distribution.py -- Stage 1: Class Distribution & Imbalance Analysis
============================================================================
Calculates exact class frequencies, percentages, and imbalance ratios across
the complete datasets without sampling.

Generates:
- Pie charts
- Bar charts
- Long-tail distribution plots

Saves:
- figures/class_distribution.png
- tables/class_distribution.csv
"""

import os
import glob
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
    filename=os.path.join(LOGS_DIR, "02_class_distribution.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

def analyze_class_distribution(name, path):
    logging.info(f"Analyzing class distribution for {name} at {path}")
    print(f"Aggregating class frequencies for {name} ...")

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

    class_counts = pd.Series(dtype=int)

    for f in files:
        print(f"  Reading {os.path.basename(f)} ...")
        peek = pd.read_csv(f, nrows=1, encoding="latin1")
        peek.columns = peek.columns.str.strip()
        
        label_col = None
        for candidate in ["Label", "label", " Label"]:
            if candidate in peek.columns:
                label_col = candidate
                break
        if label_col is None:
            label_col = peek.columns[-1]

        for chunk in pd.read_csv(f, usecols=lambda c: c.strip() == label_col, chunksize=500000, low_memory=False, encoding="latin1"):
            chunk.columns = chunk.columns.str.strip()
            series = chunk[label_col].astype(str).str.strip()
            # Filter out header repetitions
            series = series[~series.isin(["Label", "label"])]
            counts = series.value_counts()
            class_counts = class_counts.add(counts, fill_value=0)

    # Sort descending
    class_counts = class_counts.sort_values(ascending=False).astype(int)
    total_samples = class_counts.sum()
    percentages = (class_counts / total_samples) * 100
    
    imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[-1] if len(class_counts) > 1 and class_counts.iloc[-1] > 0 else 1.0

    df_stats = pd.DataFrame({
        "Dataset": name,
        "Class": class_counts.index,
        "Frequency": class_counts.values,
        "Percentage": percentages.values
    })
    
    logging.info(f"Completed class distribution for {name}. Imbalance Ratio: {imbalance_ratio:.2f}")
    return df_stats, class_counts, imbalance_ratio

def generate_visualizations(dataset_counts):
    print("\nGenerating comprehensive class distribution plots ...")
    n_datasets = len(dataset_counts)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(22, 6 * n_datasets))
    
    # Custom vibrant palettes
    palettes = ["Blues_r", "Greens_r", "Purples_r"]

    for i, (name, data) in enumerate(dataset_counts.items()):
        counts = data["counts"]
        imbalance_ratio = data["imbalance_ratio"]
        palette = palettes[i % len(palettes)]

        # Plot 1: Bar Chart (Top 10 Classes if too many)
        top_n = counts.head(10)
        ax_bar = axes[i, 0]
        sns.barplot(x=top_n.values, y=top_n.index, ax=ax_bar, palette=palette)
        ax_bar.set_title(f"{name}: Top Class Frequencies (IR: {imbalance_ratio:.1f})", fontsize=14, fontweight="bold")
        ax_bar.set_xlabel("Frequency", fontsize=12)
        ax_bar.set_ylabel("Class", fontsize=12)
        ax_bar.grid(True, axis="x", linestyle="--", alpha=0.6)

        # Plot 2: Pie Chart
        ax_pie = axes[i, 1]
        # Group small classes into 'Other' for cleaner pie chart
        if len(counts) > 6:
            pie_counts = counts.head(5).copy()
            pie_counts["Other"] = counts.iloc[5:].sum()
        else:
            pie_counts = counts
        ax_pie.pie(pie_counts.values, labels=pie_counts.index, autopct="%1.1f%%", startangle=140,
                   colors=sns.color_palette("Set2", len(pie_counts)), wedgeprops={'edgecolor': 'w', 'linewidth': 2})
        ax_pie.set_title(f"{name}: Class Percentage Breakdown", fontsize=14, fontweight="bold")

        # Plot 3: Long-Tail Distribution Plot (Log Scale)
        ax_tail = axes[i, 2]
        x_ranks = np.arange(1, len(counts) + 1)
        ax_tail.plot(x_ranks, counts.values, 'o-', color="#DC2626", linewidth=2.5, markersize=8)
        ax_tail.set_yscale("log")
        ax_tail.set_title(f"{name}: Long-Tail Distribution (Log Scale)", fontsize=14, fontweight="bold")
        ax_tail.set_xlabel("Class Rank", fontsize=12)
        ax_tail.set_ylabel("Frequency (Log)", fontsize=12)
        ax_tail.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Stage 1: Cross-Dataset Class Distribution & Imbalance Analysis", fontsize=20, fontweight="bold", y=1.02)
    sns.despine()
    plt.tight_layout()
    
    out_img = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved to {out_img}")

def main():
    print("="*60)
    print(" STAGE 1: CLASS DISTRIBUTION & IMBALANCE ANALYSIS ")
    print("="*60)
    
    all_stats = []
    dataset_counts = {}

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist -> {path}")
            continue
        df_stats, counts, ir = analyze_class_distribution(name, path)
        all_stats.append(df_stats)
        dataset_counts[name] = {"counts": counts, "imbalance_ratio": ir}

    if all_stats:
        df_full = pd.concat(all_stats, ignore_index=True)
        out_csv = os.path.join(TABLES_DIR, "class_distribution.csv")
        df_full.to_csv(out_csv, index=False)
        print(f"\nClass distribution statistics saved to {out_csv}")
        
        # Print summary
        print("\nSummary of Top Classes per Dataset:")
        for name, data in dataset_counts.items():
            print(f"\n[{name}] Total Classes: {len(data['counts'])} | Imbalance Ratio: {data['imbalance_ratio']:.2f}")
            print(data['counts'].head(5).to_string())

        generate_visualizations(dataset_counts)

    print("\nDone!")

if __name__ == "__main__":
    main()
