"""
12_tsne_visualization.py -- Stage 1: t-SNE Manifold Visualization
=================================================================
Applies t-Distributed Stochastic Neighbor Embedding (t-SNE) on a stratified
multi-dataset extraction to visualize non-linear manifold structures.

Visualizes:
- Datasets (CICIDS2017, CSE-CIC-IDS2018, Lycos-Unicas-IDS2018)
- Attack Families (BENIGN, DDoS, Bot, PortScan, etc.)

Saves:
- figures/tsne.png
"""

import os
import glob
import gc
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ─── Setup Paths & Logging ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(STAGE1_DIR, "figures")
LOGS_DIR   = os.path.join(STAGE1_DIR, "logs")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "12_tsne_visualization.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

COMMON_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Down/Up Ratio", "Average Packet Size"
]

IDS2018_MAP = {
    "Flow Duration": "Flow Duration", "Total Fwd Packets": "Tot Fwd Pkts", "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts", "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max", "Fwd Packet Length Min": "Fwd Pkt Len Min", "Fwd Packet Length Mean": "Fwd Pkt Len Mean", "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max", "Bwd Packet Length Min": "Bwd Pkt Len Min", "Bwd Packet Length Mean": "Bwd Pkt Len Mean", "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s", "Flow Packets/s": "Flow Pkts/s", "Flow IAT Mean": "Flow IAT Mean", "Flow IAT Std": "Flow IAT Std",
    "Fwd Header Length": "Fwd Header Len", "Bwd Header Length": "Bwd Header Len", "Fwd Packets/s": "Fwd Pkts/s", "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min", "Max Packet Length": "Pkt Len Max", "Packet Length Mean": "Pkt Len Mean", "Packet Length Std": "Pkt Len Std",
    "FIN Flag Count": "FIN Flag Cnt", "SYN Flag Count": "SYN Flag Cnt", "RST Flag Count": "RST Flag Cnt", "PSH Flag Count": "PSH Flag Cnt", "ACK Flag Count": "ACK Flag Cnt",
    "Down/Up Ratio": "Down/Up Ratio", "Average Packet Size": "Pkt Size Avg"
}

LYCOS_MAP = {
    "Flow Duration": "flow_duration", "Total Fwd Packets": "fwd_pkt_cnt", "Total Backward Packets": "bwd_pkt_cnt",
    "Total Length of Fwd Packets": "fwd_pkt_len_tot", "Total Length of Bwd Packets": "bwd_pkt_len_tot",
    "Fwd Packet Length Max": "fwd_pkt_len_max", "Fwd Packet Length Min": "fwd_pkt_len_min", "Fwd Packet Length Mean": "fwd_pkt_len_mean", "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max", "Bwd Packet Length Min": "bwd_pkt_len_min", "Bwd Packet Length Mean": "bwd_pkt_len_mean", "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "bytes_per_s", "Flow Packets/s": "pkt_per_s", "Flow IAT Mean": "iat_mean", "Flow IAT Std": "iat_std",
    "Fwd Header Length": "fwd_pkt_hdr_len_tot", "Bwd Header Length": "bwd_pkt_hdr_len_tot", "Fwd Packets/s": "fwd_pkt_per_s", "Bwd Packets/s": "bwd_pkt_per_s",
    "Min Packet Length": "pkt_len_min", "Max Packet Length": "pkt_len_max", "Packet Length Mean": "pkt_len_mean", "Packet Length Std": "pkt_len_std",
    "FIN Flag Count": "flag_fin", "SYN Flag Count": "flag_SYN", "RST Flag Count": "flag_rst", "PSH Flag Count": "flag_psh", "ACK Flag Count": "flag_ack",
    "Down/Up Ratio": "down_up_ratio", "Average Packet Size": "pkt_len_mean"
}

SEMANTIC_MAP = {
    "Benign": "BENIGN", "benign": "BENIGN", "BENIGN": "BENIGN",
    "Bot": "Bot", "bot": "Bot",
    "DDoS": "DDoS", "ddos": "DDoS", "DDOS attack-HOIC": "DDoS", "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS", "DDoS HOIC": "DDoS", "DDoS LOIC-HTTP": "DDoS", "DDoS LOIC-UDP": "DDoS",
    "DoS attacks-GoldenEye": "DoS", "DoS attacks-Hulk": "DoS", "DoS attacks-SlowHTTPTest": "DoS", "DoS attacks-Slowloris": "DoS",
    "DoS GoldenEye": "DoS", "DoS Hulk": "DoS", "DoS Slowhttptest": "DoS", "DoS slowloris": "DoS", "DoS Slowloris": "DoS", "DoS": "DoS",
    "PortScan": "PortScan", "Portscan": "PortScan",
    "FTP-BruteForce": "BruteForce", "FTP-Patator": "BruteForce", "SSH-Bruteforce": "BruteForce", "SSH-Patator": "BruteForce",
    "Brute Force -Web": "Web Attack", "Brute Force -XSS": "Web Attack", "SQL Injection": "Web Attack",
    "Web Attack - Brute Force": "Web Attack", "Web Attack - XSS": "Web Attack", "Web Attack - Sql Injection": "Web Attack",
    "Infilteration": "Infiltration", "Infiltration": "Infiltration", "Heartbleed": "Heartbleed"
}

def load_stratified_extract(ds_name, path):
    print(f"  Sampling stratified extract for {ds_name} ...")
    if os.path.isfile(path): files = [path]
    else: files = sorted(glob.glob(os.path.join(path, "*.csv")))

    col_map = IDS2018_MAP if ds_name == "CSE-CIC-IDS2018" else (LYCOS_MAP if ds_name == "Lycos-Unicas-IDS2018" else {f: f for f in COMMON_FEATURES})
    target_cols = [col_map.get(f, f) for f in COMMON_FEATURES]

    chunks = []
    total_samples = 0
    # 8,000 samples per dataset (24,000 total) ensures t-SNE completes within 2 minutes while capturing beautiful manifold structures
    max_samples = 8000 

    for f in files:
        try:
            for chunk in pd.read_csv(f, chunksize=500000, low_memory=False, encoding="latin1"):
                chunk.columns = chunk.columns.str.strip()
                label_col = None
                for candidate in ["Label", "label", " Label"]:
                    if candidate in chunk.columns: label_col = candidate; break
                if label_col is None: label_col = chunk.columns[-1]

                avail_cols = [c for c in target_cols if c in chunk.columns]
                if not avail_cols: continue
                
                sub = chunk[avail_cols + [label_col]].copy()
                inv_map = {v: k for k, v in col_map.items()}
                sub = sub.rename(columns=inv_map)
                
                # Standardize label to Attack Family
                raw_labels = sub[label_col].astype(str).str.strip()
                sub["Attack Family"] = raw_labels.map(SEMANTIC_MAP).fillna("Other")
                sub = sub[sub["Attack Family"] != "Label"] # Drop header repetitions
                
                for feat in COMMON_FEATURES:
                    if feat not in sub.columns: sub[feat] = 0.0
                
                # Clean numeric
                for c in COMMON_FEATURES:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)
                sub.replace([np.inf, -np.inf], 0.0, inplace=True)
                
                # Stratified sample by Attack Family
                strat_sample = sub.sample(frac=1, random_state=42).groupby("Attack Family").head(800)
                chunks.append(strat_sample)
                total_samples += len(strat_sample)
                if total_samples >= max_samples: break
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
            continue
        if total_samples >= max_samples: break

    if not chunks: return pd.DataFrame()
    df_full = pd.concat(chunks, ignore_index=True)
    df_full["Dataset"] = ds_name
    if len(df_full) > max_samples:
        n_per_group = max_samples // max(1, df_full["Attack Family"].nunique())
        df_full = df_full.sample(frac=1, random_state=42).groupby("Attack Family").head(n_per_group)
    return df_full

def main():
    print("="*60)
    print(" STAGE 1: t-SNE MANIFOLD VISUALIZATION ")
    print("="*60)
    
    dfs = []
    for ds_name, path in DATASETS.items():
        if os.path.exists(path):
            df = load_stratified_extract(ds_name, path)
            if not df.empty:
                dfs.append(df)
                
    if not dfs:
        print("ERROR: No dataset matrices loaded for t-SNE.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    X = df_all[COMMON_FEATURES].values
    y_dataset = df_all["Dataset"].values
    y_family  = df_all["Attack Family"].values
    
    print(f"\nFitting StandardScaler on {len(df_all)} samples ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Fitting TSNE(n_components=2, perplexity=35, max_iter=1000) ... (This may take 1-2 minutes)")
    tsne = TSNE(n_components=2, perplexity=35, max_iter=1000, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame({"t-SNE 1": X_tsne[:, 0], "t-SNE 2": X_tsne[:, 1], "Dataset": y_dataset, "Attack Family": y_family})
    
    print("\nGenerating t-SNE subplots ...")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    # Subplot 1: Colored by Dataset
    palette_ds = {"CICIDS2017": "#2563EB", "CSE-CIC-IDS2018": "#10B981", "Lycos-Unicas-IDS2018": "#D97706"}
    sns.scatterplot(data=df_tsne, x="t-SNE 1", y="t-SNE 2", hue="Dataset", palette=palette_ds,
                    alpha=0.6, s=25, edgecolor=None, ax=axes[0])
    axes[0].set_title("t-SNE Manifold: Dataset Distribution & Domain Shift", fontsize=16, fontweight="bold", pad=15)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Colored by Attack Family
    n_families = df_tsne["Attack Family"].nunique()
    palette_fam = sns.color_palette("tab10", n_families)
    sns.scatterplot(data=df_tsne, x="t-SNE 1", y="t-SNE 2", hue="Attack Family", palette=palette_fam,
                    alpha=0.7, s=25, edgecolor=None, ax=axes[1])
    axes[1].set_title("t-SNE Manifold: Attack Family Clusters & Structural Alignment", fontsize=16, fontweight="bold", pad=15)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Attack Family", fontsize=11, title_fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Stage 1: High-Dimensional Manifold Visualization (t-SNE)", fontsize=22, fontweight="bold", y=1.05)
    sns.despine()
    plt.tight_layout()
    
    out_img = os.path.join(FIGURES_DIR, "tsne.png")
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"t-SNE visualization saved to {out_img}")
    print("\nDone!")

if __name__ == "__main__":
    main()
