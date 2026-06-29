"""
10_dataset_shift_metrics.py -- Stage 1: Mathematical Dataset Shift Metrics
==========================================================================
Calculates advanced distribution divergence metrics for every shared feature:
- KL divergence
- Jensen-Shannon divergence
- Wasserstein distance

Compares:
- CICIDS2017 vs CSE-CIC-IDS2018
- CICIDS2017 vs Lycos-Unicas-IDS2018

Ranks most shifted and least shifted features.

Saves:
- tables/dataset_shift_scores.csv
- figures/feature_drift_ranking.png
"""

import os
import glob
import gc
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
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
    filename=os.path.join(LOGS_DIR, "10_dataset_shift_metrics.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

# 80 features from CICIDS2017 training
CICIDS2017_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

IDS2018_MAP = {
    "Flow Duration": "Flow Duration", "Total Fwd Packets": "Tot Fwd Pkts", "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts", "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max", "Fwd Packet Length Min": "Fwd Pkt Len Min", "Fwd Packet Length Mean": "Fwd Pkt Len Mean", "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max", "Bwd Packet Length Min": "Bwd Pkt Len Min", "Bwd Packet Length Mean": "Bwd Pkt Len Mean", "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s", "Flow Packets/s": "Flow Pkts/s",
    "Flow IAT Mean": "Flow IAT Mean", "Flow IAT Std": "Flow IAT Std", "Flow IAT Max": "Flow IAT Max", "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Total": "Fwd IAT Tot", "Fwd IAT Mean": "Fwd IAT Mean", "Fwd IAT Std": "Fwd IAT Std", "Fwd IAT Max": "Fwd IAT Max", "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Total": "Bwd IAT Tot", "Bwd IAT Mean": "Bwd IAT Mean", "Bwd IAT Std": "Bwd IAT Std", "Bwd IAT Max": "Bwd IAT Max", "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags", "Bwd PSH Flags": "Bwd PSH Flags", "Fwd URG Flags": "Fwd URG Flags", "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Length": "Fwd Header Len", "Bwd Header Length": "Bwd Header Len", "Fwd Packets/s": "Fwd Pkts/s", "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min", "Max Packet Length": "Pkt Len Max", "Packet Length Mean": "Pkt Len Mean", "Packet Length Std": "Pkt Len Std", "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt", "SYN Flag Count": "SYN Flag Cnt", "RST Flag Count": "RST Flag Cnt", "PSH Flag Count": "PSH Flag Cnt", "ACK Flag Count": "ACK Flag Cnt", "URG Flag Count": "URG Flag Cnt",
    "CWE Flag Count": "CWE Flag Count", "ECE Flag Count": "ECE Flag Cnt", "Down/Up Ratio": "Down/Up Ratio",
    "Average Packet Size": "Pkt Size Avg", "Avg Fwd Segment Size": "Fwd Seg Size Avg", "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg", "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg", "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg", "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg", "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts", "Subflow Fwd Bytes": "Subflow Fwd Byts", "Subflow Bwd Packets": "Subflow Bwd Pkts", "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts", "Init_Win_bytes_backward": "Init Bwd Win Byts", "act_data_pkt_fwd": "Fwd Act Data Pkts", "min_seg_size_forward": "Fwd Seg Size Min",
    "Active Mean": "Active Mean", "Active Std": "Active Std", "Active Max": "Active Max", "Active Min": "Active Min",
    "Idle Mean": "Idle Mean", "Idle Std": "Idle Std", "Idle Max": "Idle Max", "Idle Min": "Idle Min",
}

LYCOS_MAP = {
    "Flow Duration": "flow_duration", "Total Fwd Packets": "fwd_pkt_cnt", "Total Backward Packets": "bwd_pkt_cnt",
    "Total Length of Fwd Packets": "fwd_pkt_len_tot", "Total Length of Bwd Packets": "bwd_pkt_len_tot",
    "Fwd Packet Length Max": "fwd_pkt_len_max", "Fwd Packet Length Min": "fwd_pkt_len_min", "Fwd Packet Length Mean": "fwd_pkt_len_mean", "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max", "Bwd Packet Length Min": "bwd_pkt_len_min", "Bwd Packet Length Mean": "bwd_pkt_len_mean", "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "bytes_per_s", "Flow Packets/s": "pkt_per_s",
    "Flow IAT Mean": "iat_mean", "Flow IAT Std": "iat_std", "Flow IAT Max": "iat_max", "Flow IAT Min": "iat_min",
    "Fwd IAT Total": "fwd_iat_tot", "Fwd IAT Mean": "fwd_iat_mean", "Fwd IAT Std": "fwd_iat_std", "Fwd IAT Max": "fwd_iat_max", "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot", "Bwd IAT Mean": "bwd_iat_mean", "Bwd IAT Std": "bwd_iat_std", "Bwd IAT Max": "bwd_iat_max", "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_flag_psh", "Bwd PSH Flags": "bwd_flag_psh", "Fwd URG Flags": "fwd_flag_urg", "Bwd URG Flags": "bwd_flag_urg",
    "Fwd Header Length": "fwd_pkt_hdr_len_tot", "Bwd Header Length": "bwd_pkt_hdr_len_tot", "Fwd Packets/s": "fwd_pkt_per_s", "Bwd Packets/s": "bwd_pkt_per_s",
    "Min Packet Length": "pkt_len_min", "Max Packet Length": "pkt_len_max", "Packet Length Mean": "pkt_len_mean", "Packet Length Std": "pkt_len_std", "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "flag_fin", "SYN Flag Count": "flag_SYN", "RST Flag Count": "flag_rst", "PSH Flag Count": "flag_psh", "ACK Flag Count": "flag_ack", "URG Flag Count": "flag_urg",
    "CWE Flag Count": "flag_cwr", "ECE Flag Count": "flag_ece", "Down/Up Ratio": "down_up_ratio",
    "Fwd Avg Bytes/Bulk": "fwd_bulk_bytes_mean", "Fwd Avg Packets/Bulk": "fwd_bulk_pkt_mean", "Fwd Avg Bulk Rate": "fwd_bulk_rate_mean",
    "Bwd Avg Bytes/Bulk": "bwd_bulk_bytes_mean", "Bwd Avg Packets/Bulk": "bwd_bulk_pkt_mean", "Bwd Avg Bulk Rate": "bwd_bulk_rate_mean",
    "Subflow Fwd Packets": "fwd_subflow_pkt_mean", "Subflow Fwd Bytes": "fwd_subflow_bytes_mean", "Subflow Bwd Packets": "bwd_subflow_pkt_mean", "Subflow Bwd Bytes": "bwd_subflow_bytes_mean",
    "Init_Win_bytes_forward": "fwd_tcp_init_win_bytes", "Init_Win_bytes_backward": "bwd_tcp_init_win_bytes",
    "act_data_pkt_fwd": "fwd_non_empty_pkt_cnt", "min_seg_size_forward": "fwd_pkt_hdr_len_min",
    "Active Mean": "active_mean", "Active Std": "active_std", "Active Max": "active_max", "Active Min": "active_min",
    "Idle Mean": "idle_mean", "Idle Std": "idle_std", "Idle Max": "idle_max", "Idle Min": "idle_min",
}

def load_feature_series(dataset_name, path, std_feat_name):
    if dataset_name == "CICIDS2017": col = std_feat_name
    elif dataset_name == "CSE-CIC-IDS2018": col = IDS2018_MAP.get(std_feat_name, std_feat_name)
    elif dataset_name == "Lycos-Unicas-IDS2018": col = LYCOS_MAP.get(std_feat_name, std_feat_name)

    if os.path.isfile(path): files = [path]
    else: files = sorted(glob.glob(os.path.join(path, "*.csv")))

    chunks = []
    for f in files:
        try:
            for chunk in pd.read_csv(f, usecols=lambda c: c.strip() == col, chunksize=1000000, low_memory=False, encoding="latin1"):
                chunk.columns = chunk.columns.str.strip()
                s = pd.to_numeric(chunk[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) > 20000: s = s.sample(20000, random_state=42)
                chunks.append(s)
        except Exception as e:
            logging.error(f"Error reading {col} in {f}: {e}")
            continue

    if not chunks: return np.array([])
    full_s = pd.concat(chunks, ignore_index=True)
    if len(full_s) > 100000: full_s = full_s.sample(100000, random_state=42)
    return full_s.values

def calculate_shift_metrics(u, v):
    if len(u) == 0 or len(v) == 0:
        return np.nan, np.nan, np.nan

    # Wasserstein distance
    wd = stats.wasserstein_distance(u, v)

    # KL and JS divergence require common histogram bins
    min_val = min(np.min(u), np.min(v))
    max_val = max(np.max(u), np.max(v))
    
    if min_val == max_val:
        return 0.0, 0.0, 0.0

    bins = np.linspace(min_val, max_val, 50)
    p, _ = np.histogram(u, bins=bins, density=True)
    q, _ = np.histogram(v, bins=bins, density=True)

    # Epsilon smoothing to prevent log(0)
    eps = 1e-9
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)

    kl = np.sum(p * np.log(p / q))
    js = jensenshannon(p, q) ** 2  # JS divergence is square of JS distance

    return kl, js, wd

def main():
    print("="*60)
    print(" STAGE 1: MATHEMATICAL DATASET SHIFT METRICS ")
    print("="*60)
    
    results = []
    
    path_2017 = DATASETS.get("CICIDS2017")
    path_2018 = DATASETS.get("CSE-CIC-IDS2018")
    path_lycos = DATASETS.get("Lycos-Unicas-IDS2018")
    
    if not os.path.exists(path_2017):
        print("ERROR: CICIDS2017 baseline dataset not found.")
        return

    print("Iterating through shared features to calculate divergence scores ...")
    for feat in CICIDS2017_FEATURES:
        print(f"  Evaluating shift for feature: {feat} ...")
        u_2017 = load_feature_series("CICIDS2017", path_2017, feat)
        
        if len(u_2017) == 0:
            continue
            
        # CICIDS2017 vs CSE-CIC-IDS2018
        if os.path.exists(path_2018):
            v_2018 = load_feature_series("CSE-CIC-IDS2018", path_2018, feat)
            kl_18, js_18, wd_18 = calculate_shift_metrics(u_2017, v_2018)
        else:
            kl_18, js_18, wd_18 = np.nan, np.nan, np.nan
            
        # CICIDS2017 vs Lycos
        if os.path.exists(path_lycos):
            v_lycos = load_feature_series("Lycos-Unicas-IDS2018", path_lycos, feat)
            kl_lycos, js_lycos, wd_lycos = calculate_shift_metrics(u_2017, v_lycos)
        else:
            kl_lycos, js_lycos, wd_lycos = np.nan, np.nan, np.nan
            
        results.append({
            "Feature": feat,
            "KL_2018": kl_18, "JS_2018": js_18, "Wasserstein_2018": wd_18,
            "KL_Lycos": kl_lycos, "JS_Lycos": js_lycos, "Wasserstein_Lycos": wd_lycos,
            # Average JS divergence as a stable ranking metric (bounded [0, 1])
            "Mean_JS_Drift": np.mean([js_18, js_lycos]) if not np.isnan(js_18) and not np.isnan(js_lycos) else (js_18 if not np.isnan(js_18) else js_lycos)
        })
        
        gc.collect()

    df_shift = pd.DataFrame(results)
    df_shift = df_shift.sort_values(by="Mean_JS_Drift", ascending=False).reset_index(drop=True)
    
    out_csv = os.path.join(TABLES_DIR, "dataset_shift_scores.csv")
    df_shift.to_csv(out_csv, index=False)
    print(f"\nDataset shift scores saved to {out_csv}")
    
    print("\nTop 10 Most Shifted Features (Ranked by Mean JS Divergence):")
    print(df_shift.head(10)[["Feature", "Mean_JS_Drift", "JS_2018", "JS_Lycos", "Wasserstein_2018", "Wasserstein_Lycos"]].to_string(index=False))
    
    print("\nTop 5 Least Shifted Features:")
    print(df_shift.tail(5)[["Feature", "Mean_JS_Drift", "JS_2018", "JS_Lycos", "Wasserstein_2018", "Wasserstein_Lycos"]].to_string(index=False))

    # Generate Figures
    print("\nGenerating feature drift ranking bar chart ...")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    top_30 = df_shift.head(30)
    sns.barplot(data=top_30, x="Mean_JS_Drift", y="Feature", ax=ax, palette="magma")
    ax.set_title("Stage 1: Top 30 Most Shifted Features (Ranked by Jensen-Shannon Divergence)", fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Mean Jensen-Shannon Divergence (Drift Score)", fontsize=14)
    ax.set_ylabel("Feature Name", fontsize=14)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    
    sns.despine()
    plt.tight_layout()
    
    out_img = os.path.join(FIGURES_DIR, "feature_drift_ranking.png")
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved to {out_img}")
    print("\nDone!")

if __name__ == "__main__":
    main()
