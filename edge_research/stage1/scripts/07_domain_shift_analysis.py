"""
07_domain_shift_analysis.py -- Stage 1: Empirical Domain Shift Analysis
=======================================================================
Compares feature distributions across CICIDS2017, CSE-CIC-IDS2018, and Lycos.

Generates comparative visualizations for important features:
- Flow Duration
- Packet Length Mean
- Packet Length Std
- Total Fwd Packets
- Total Backward Packets
Includes histograms, KDE plots, and boxplots (log-scaled for network traffic).

Saves:
- tables/domain_shift_report.csv
- figures/domain_shift.png
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
    filename=os.path.join(LOGS_DIR, "07_domain_shift_analysis.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

# Mapping dictionaries
IDS2018_MAP = {
    "Flow Duration": "Flow Duration",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts"
}

LYCOS_MAP = {
    "Flow Duration": "flow_duration",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Total Fwd Packets": "fwd_pkt_cnt",
    "Total Backward Packets": "bwd_pkt_cnt"
}

IMPORTANT_FEATURES = [
    "Flow Duration",
    "Packet Length Mean",
    "Packet Length Std",
    "Total Fwd Packets",
    "Total Backward Packets"
]

def load_feature_series(dataset_name, path, std_feat_name):
    if dataset_name == "CICIDS2017":
        col = std_feat_name
    elif dataset_name == "CSE-CIC-IDS2018":
        col = IDS2018_MAP.get(std_feat_name, std_feat_name)
    elif dataset_name == "Lycos-Unicas-IDS2018":
        col = LYCOS_MAP.get(std_feat_name, std_feat_name)

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")))

    chunks = []
    for f in files:
        try:
            for chunk in pd.read_csv(f, usecols=lambda c: c.strip() == col, chunksize=1000000, low_memory=False, encoding="latin1"):
                chunk.columns = chunk.columns.str.strip()
                s = pd.to_numeric(chunk[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                # Take a robust uniform sample from each chunk if extremely large to make KDE plotting tractable
                if len(s) > 50000:
                    s = s.sample(50000, random_state=42)
                chunks.append(s)
        except Exception as e:
            logging.error(f"Error reading {col} in {f}: {e}")
            continue

    if not chunks:
        return pd.Series(dtype=float)
    
    full_s = pd.concat(chunks, ignore_index=True)
    if len(full_s) > 300000: # Cap at 300k per dataset for KDE plot performance
        full_s = full_s.sample(300000, random_state=42)
    return full_s

def build_domain_shift_report():
    stats_csv = os.path.join(TABLES_DIR, "feature_statistics.csv")
    if not os.path.exists(stats_csv):
        print(f"ERROR: {stats_csv} not found. Run 06_feature_statistics.py first.")
        return

    df_stats = pd.read_csv(stats_csv)
    
    # Map external feature names back to standard CICIDS2017 names for merging
    inv_2018 = {v: k for k, v in IDS2018_MAP.items()}
    inv_lycos = {v: k for k, v in LYCOS_MAP.items()}
    
    # Map back standard names
    def standardize_feat(row):
        ds = row["Dataset"]
        feat = row["Feature"]
        if ds == "CSE-CIC-IDS2018" and feat in inv_2018:
            return inv_2018[feat]
        if ds == "Lycos-Unicas-IDS2018" and feat in inv_lycos:
            return inv_lycos[feat]
        return feat

    df_stats["Std_Feature"] = df_stats.apply(standardize_feat, axis=1)
    
    # Pivot table to compare mean and std across datasets
    pivot_mean = df_stats.pivot_table(index="Std_Feature", columns="Dataset", values="mean", aggfunc="first")
    pivot_std  = df_stats.pivot_table(index="Std_Feature", columns="Dataset", values="std", aggfunc="first")
    
    df_report = pd.DataFrame(index=pivot_mean.index)
    df_report["CIC2017_Mean"] = pivot_mean.get("CICIDS2017", np.nan)
    df_report["CIC2018_Mean"] = pivot_mean.get("CSE-CIC-IDS2018", np.nan)
    df_report["Lycos_Mean"]   = pivot_mean.get("Lycos-Unicas-IDS2018", np.nan)
    
    df_report["CIC2017_Std"] = pivot_std.get("CICIDS2017", np.nan)
    df_report["CIC2018_Std"] = pivot_std.get("CSE-CIC-IDS2018", np.nan)
    df_report["Lycos_Std"]   = pivot_std.get("Lycos-Unicas-IDS2018", np.nan)
    
    # Compute relative shift (Diff in mean / Train Std)
    df_report["Shift_2018 (Z-score)"] = np.abs(df_report["CIC2018_Mean"] - df_report["CIC2017_Mean"]) / (df_report["CIC2017_Std"] + 1e-5)
    df_report["Shift_Lycos (Z-score)"] = np.abs(df_report["Lycos_Mean"] - df_report["CIC2017_Mean"]) / (df_report["CIC2017_Std"] + 1e-5)
    
    df_report = df_report.sort_values(by="Shift_Lycos (Z-score)", ascending=False).reset_index()
    out_csv = os.path.join(TABLES_DIR, "domain_shift_report.csv")
    df_report.to_csv(out_csv, index=False)
    print(f"\nDomain shift report saved to {out_csv}")
    print("\nTop Shifted Features Summary:")
    print(df_report.head(10).to_string(index=False))

def generate_shift_plots():
    print("\nGenerating empirical domain shift plots (Histograms, KDEs, Boxplots) ...")
    fig, axes = plt.subplots(len(IMPORTANT_FEATURES), 3, figsize=(24, 5 * len(IMPORTANT_FEATURES)))
    
    palette = {"CICIDS2017": "#2563EB", "CSE-CIC-IDS2018": "#10B981", "Lycos-Unicas-IDS2018": "#D97706"}

    for i, feat in enumerate(IMPORTANT_FEATURES):
        print(f"  Plotting feature: {feat} ...")
        df_feat_list = []
        for ds_name, path in DATASETS.items():
            if os.path.exists(path):
                s = load_feature_series(ds_name, path, feat)
                if not s.empty:
                    # Log transform for clear visualization of highly skewed network data (e.g. Flow Duration)
                    s_log = np.log1p(np.clip(s.values, 0, None))
                    df_feat_list.append(pd.DataFrame({"Dataset": ds_name, "Value (Log1p)": s_log}))
                    
        if not df_feat_list:
            continue
            
        df_plot = pd.concat(df_feat_list, ignore_index=True)
        
        # Plot 1: Histogram
        ax_hist = axes[i, 0]
        sns.histplot(data=df_plot, x="Value (Log1p)", hue="Dataset", stat="density", common_norm=False,
                     bins=40, alpha=0.5, ax=ax_hist, palette=palette, element="step")
        ax_hist.set_title(f"{feat}: Distribution Comparison (Histogram)", fontsize=14, fontweight="bold")
        ax_hist.set_xlabel(f"{feat} (Log1p Scale)", fontsize=12)
        ax_hist.grid(True, linestyle="--", alpha=0.6)
        
        # Plot 2: KDE Plot
        ax_kde = axes[i, 1]
        sns.kdeplot(data=df_plot, x="Value (Log1p)", hue="Dataset", common_norm=False,
                    fill=True, alpha=0.3, linewidth=2.5, ax=ax_kde, palette=palette)
        ax_kde.set_title(f"{feat}: Density Estimation (KDE)", fontsize=14, fontweight="bold")
        ax_kde.set_xlabel(f"{feat} (Log1p Scale)", fontsize=12)
        ax_kde.grid(True, linestyle="--", alpha=0.6)
        
        # Plot 3: Boxplot
        ax_box = axes[i, 2]
        sns.boxplot(data=df_plot, x="Dataset", y="Value (Log1p)", ax=ax_box, palette=palette, width=0.5, fliersize=1.5)
        ax_box.set_title(f"{feat}: Quartile & Outlier Comparison (Boxplot)", fontsize=14, fontweight="bold")
        ax_box.set_xlabel("Dataset", fontsize=12)
        ax_box.set_ylabel(f"{feat} (Log1p Scale)", fontsize=12)
        ax_box.grid(True, axis="y", linestyle="--", alpha=0.6)
        
        del df_plot, df_feat_list
        gc.collect()

    plt.suptitle("Stage 1: Cross-Dataset Empirical Domain Shift Analysis", fontsize=22, fontweight="bold", y=1.02)
    sns.despine()
    plt.tight_layout()
    
    out_img = os.path.join(FIGURES_DIR, "domain_shift.png")
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved to {out_img}")

def main():
    print("="*60)
    print(" STAGE 1: EMPIRICAL DOMAIN SHIFT ANALYSIS ")
    print("="*60)
    
    build_domain_shift_report()
    generate_shift_plots()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
