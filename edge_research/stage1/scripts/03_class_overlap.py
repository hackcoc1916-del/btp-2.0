"""
03_class_overlap.py -- Stage 1: Class Overlap & Unseen Attack Analysis
======================================================================
Determines common, unique, and missing attack classes between the training
dataset (CICIDS2017) and evaluation datasets (CSE-CIC-IDS2018, Lycos).

Creates table with columns:
- TRAIN CLASS
- TEST CLASS
- COMMON
- UNSEEN

Saves:
- tables/class_overlap.csv
- figures/class_overlap.png (Custom Venn diagrams & overlap analysis)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    filename=os.path.join(LOGS_DIR, "03_class_overlap.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

# High-level semantic mapping to group exact/similar attack concepts
SEMANTIC_MAP = {
    "Benign": "BENIGN", "benign": "BENIGN", "BENIGN": "BENIGN",
    "Bot": "Bot", "bot": "Bot",
    "DDoS": "DDoS", "ddos": "DDoS", "DDOS attack-HOIC": "DDoS", "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS", "DDoS HOIC": "DDoS", "DDoS LOIC-HTTP": "DDoS", "DDoS LOIC-UDP": "DDoS",
    "DoS attacks-GoldenEye": "DoS GoldenEye", "DoS GoldenEye": "DoS GoldenEye",
    "DoS attacks-Hulk": "DoS Hulk", "DoS Hulk": "DoS Hulk",
    "DoS attacks-SlowHTTPTest": "DoS Slowhttptest", "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS attacks-Slowloris": "DoS slowloris", "DoS slowloris": "DoS slowloris", "DoS Slowloris": "DoS slowloris",
    "PortScan": "PortScan", "Portscan": "PortScan",
    "FTP-BruteForce": "FTP-Patator", "FTP-Patator": "FTP-Patator",
    "SSH-Bruteforce": "SSH-Patator", "SSH-Patator": "SSH-Patator",
    "Brute Force -Web": "Web Attack - Brute Force", "Web Attack - Brute Force": "Web Attack - Brute Force",
    "Brute Force -XSS": "Web Attack - XSS", "Web Attack - XSS": "Web Attack - XSS",
    "SQL Injection": "Web Attack - Sql Injection", "Web Attack - Sql Injection": "Web Attack - Sql Injection",
    "Infilteration": "Infiltration", "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed"
}

def get_unique_classes(name, path):
    print(f"Scanning unique classes for {name} ...")
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

    unique_classes = set()
    for f in files:
        peek = pd.read_csv(f, nrows=1, encoding="latin1")
        peek.columns = peek.columns.str.strip()
        label_col = None
        for candidate in ["Label", "label", " Label"]:
            if candidate in peek.columns:
                label_col = candidate
                break
        if label_col is None: label_col = peek.columns[-1]

        for chunk in pd.read_csv(f, usecols=lambda c: c.strip() == label_col, chunksize=500000, low_memory=False, encoding="latin1"):
            chunk.columns = chunk.columns.str.strip()
            vals = chunk[label_col].astype(str).str.strip().unique()
            unique_classes.update([val for val in vals if val not in ["Label", "label"]])
    return sorted([str(x) for x in unique_classes if str(x) != 'nan'])

def draw_custom_venn(ax, set_a, set_b, label_a, label_b, title):
    # Draw beautiful custom Venn diagram using matplotlib circles
    only_a = set_a - set_b
    only_b = set_b - set_a
    common = set_a & set_b
    
    # Circles
    circle_a = patches.Circle((0.35, 0.5), 0.28, facecolor='#3B82F6', alpha=0.5, edgecolor='#1E3A8A', linewidth=2)
    circle_b = patches.Circle((0.65, 0.5), 0.28, facecolor='#10B981', alpha=0.5, edgecolor='#065F46', linewidth=2)
    
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Labels
    ax.text(0.20, 0.82, label_a, fontsize=13, fontweight='bold', color='#1E3A8A', ha='center')
    ax.text(0.80, 0.82, label_b, fontsize=13, fontweight='bold', color='#065F46', ha='center')
    
    # Counts & items
    ax.text(0.22, 0.5, f"Unique\n({len(only_a)})\n\n" + "\n".join(list(only_a)[:4]), fontsize=10, ha='center', va='center', color='black')
    ax.text(0.78, 0.5, f"Unseen\n({len(only_b)})\n\n" + "\n".join(list(only_b)[:4]), fontsize=10, ha='center', va='center', color='black')
    ax.text(0.50, 0.5, f"Common\n({len(common)})\n\n" + "\n".join(list(common)[:4]), fontsize=10, ha='center', va='center', color='black', fontweight='bold')
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

def main():
    print("="*60)
    print(" STAGE 1: CLASS OVERLAP & UNSEEN ATTACK ANALYSIS ")
    print("="*60)
    
    # Load class distribution if available, otherwise scan
    dist_csv = os.path.join(TABLES_DIR, "class_distribution.csv")
    if os.path.exists(dist_csv):
        print("Loading classes from class_distribution.csv ...")
        df_dist = pd.read_csv(dist_csv)
        dataset_classes = {ds: df_dist[df_dist["Dataset"] == ds]["Class"].tolist() for ds in df_dist["Dataset"].unique()}
    else:
        dataset_classes = {}
        for name, path in DATASETS.items():
            if os.path.exists(path):
                dataset_classes[name] = get_unique_classes(name, path)
                
    train_classes_raw = dataset_classes.get("CICIDS2017", [])
    train_classes_sem = set(SEMANTIC_MAP.get(c, c) for c in train_classes_raw)

    results = []
    
    # Process CSE-CIC-IDS2018
    ids2018_raw = dataset_classes.get("CSE-CIC-IDS2018", [])
    for c in ids2018_raw:
        sem = SEMANTIC_MAP.get(c, c)
        is_common = sem in train_classes_sem
        train_match = sem if is_common else "N/A (Unseen)"
        results.append({
            "TEST DATASET": "CSE-CIC-IDS2018",
            "TRAIN CLASS": train_match,
            "TEST CLASS": c,
            "COMMON": "YES" if is_common else "NO",
            "UNSEEN": "YES" if not is_common else "NO"
        })
        
    # Process Lycos-Unicas-IDS2018
    lycos_raw = dataset_classes.get("Lycos-Unicas-IDS2018", [])
    for c in lycos_raw:
        sem = SEMANTIC_MAP.get(c, c)
        is_common = sem in train_classes_sem
        train_match = sem if is_common else "N/A (Unseen)"
        results.append({
            "TEST DATASET": "Lycos-Unicas-IDS2018",
            "TRAIN CLASS": train_match,
            "TEST CLASS": c,
            "COMMON": "YES" if is_common else "NO",
            "UNSEEN": "YES" if not is_common else "NO"
        })

    df_overlap = pd.DataFrame(results)
    out_csv = os.path.join(TABLES_DIR, "class_overlap.csv")
    df_overlap.to_csv(out_csv, index=False)
    print(f"\nClass overlap table saved to {out_csv}")
    
    print("\nClass Overlap Summary Table:")
    print(df_overlap.to_string(index=False))

    # Generate Figures
    print("\nGenerating Venn diagrams & overlap plots ...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    ids2018_sem = set(SEMANTIC_MAP.get(c, c) for c in ids2018_raw)
    lycos_sem   = set(SEMANTIC_MAP.get(c, c) for c in lycos_raw)

    draw_custom_venn(axes[0], train_classes_sem, ids2018_sem, "CICIDS2017 (Train)", "CSE-CIC-IDS2018 (Test)", "CICIDS2017 vs CSE-CIC-IDS2018 Overlap")
    draw_custom_venn(axes[1], train_classes_sem, lycos_sem, "CICIDS2017 (Train)", "Lycos-Unicas-IDS2018 (Test)", "CICIDS2017 vs Lycos Overlap")

    plt.suptitle("Stage 1: Attack Class Overlap & Unseen Threat Generalization", fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_img = os.path.join(FIGURES_DIR, "class_overlap.png")
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved to {out_img}")
    print("\nDone!")

if __name__ == "__main__":
    main()
