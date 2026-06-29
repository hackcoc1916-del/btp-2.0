"""
04_feature_inventory.py -- Stage 1: Feature Inventory & Mapping Audit
=====================================================================
Determines missing features, extra features, renamed features, and different
feature counts across CICIDS2017, CSE-CIC-IDS2018, and Lycos-Unicas-IDS2018.

Generates table with columns:
- Feature
- CIC2017
- CIC2018
- Lycos
- Status

Saves:
- tables/feature_inventory.csv
- figures/feature_overlap.png
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
    filename=os.path.join(LOGS_DIR, "04_feature_inventory.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

# Original 80 features from CICIDS2017 training
CICIDS2017_FEATURES = [
    "Source Port", "Destination Port", "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

IDS2018_TO_CICIDS2017 = {
    "Dst Port": "Destination Port", "Protocol": "Protocol", "Flow Duration": "Flow Duration",
    "Tot Fwd Pkts": "Total Fwd Packets", "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets", "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max", "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean", "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max", "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean", "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s", "Flow Pkts/s": "Flow Packets/s",
    "Flow IAT Mean": "Flow IAT Mean", "Flow IAT Std": "Flow IAT Std", "Flow IAT Max": "Flow IAT Max", "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Tot": "Fwd IAT Total", "Fwd IAT Mean": "Fwd IAT Mean", "Fwd IAT Std": "Fwd IAT Std", "Fwd IAT Max": "Fwd IAT Max", "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Tot": "Bwd IAT Total", "Bwd IAT Mean": "Bwd IAT Mean", "Bwd IAT Std": "Bwd IAT Std", "Bwd IAT Max": "Bwd IAT Max", "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags", "Bwd PSH Flags": "Bwd PSH Flags", "Fwd URG Flags": "Fwd URG Flags", "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Len": "Fwd Header Length", "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s", "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length", "Pkt Len Max": "Max Packet Length", "Pkt Len Mean": "Packet Length Mean", "Pkt Len Std": "Packet Length Std", "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count", "SYN Flag Cnt": "SYN Flag Count", "RST Flag Cnt": "RST Flag Count", "PSH Flag Cnt": "PSH Flag Count", "ACK Flag Cnt": "ACK Flag Count", "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Count": "CWE Flag Count", "ECE Flag Cnt": "ECE Flag Count", "Down/Up Ratio": "Down/Up Ratio",
    "Pkt Size Avg": "Average Packet Size", "Fwd Seg Size Avg": "Avg Fwd Segment Size", "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk", "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk", "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk", "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk", "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets", "Subflow Fwd Byts": "Subflow Fwd Bytes", "Subflow Bwd Pkts": "Subflow Bwd Packets", "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward", "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd", "Fwd Seg Size Min": "min_seg_size_forward",
    "Active Mean": "Active Mean", "Active Std": "Active Std", "Active Max": "Active Max", "Active Min": "Active Min",
    "Idle Mean": "Idle Mean", "Idle Std": "Idle Std", "Idle Max": "Idle Max", "Idle Min": "Idle Min",
}

LYCOS_TO_CICIDS2017 = {
    "dst_port": "Destination Port", "ip_prot": "Protocol", "flow_duration": "Flow Duration",
    "fwd_pkt_cnt": "Total Fwd Packets", "bwd_pkt_cnt": "Total Backward Packets",
    "fwd_pkt_len_tot": "Total Length of Fwd Packets", "bwd_pkt_len_tot": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max", "fwd_pkt_len_min": "Fwd Packet Length Min", "fwd_pkt_len_mean": "Fwd Packet Length Mean", "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max", "bwd_pkt_len_min": "Bwd Packet Length Min", "bwd_pkt_len_mean": "Bwd Packet Length Mean", "bwd_pkt_len_std": "Bwd Packet Length Std",
    "bytes_per_s": "Flow Bytes/s", "pkt_per_s": "Flow Packets/s",
    "iat_mean": "Flow IAT Mean", "iat_std": "Flow IAT Std", "iat_max": "Flow IAT Max", "iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total", "fwd_iat_mean": "Fwd IAT Mean", "fwd_iat_std": "Fwd IAT Std", "fwd_iat_max": "Fwd IAT Max", "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total", "bwd_iat_mean": "Bwd IAT Mean", "bwd_iat_std": "Bwd IAT Std", "bwd_iat_max": "Bwd IAT Max", "bwd_iat_min": "Bwd IAT Min",
    "fwd_flag_psh": "Fwd PSH Flags", "bwd_flag_psh": "Bwd PSH Flags", "fwd_flag_urg": "Fwd URG Flags", "bwd_flag_urg": "Bwd URG Flags",
    "fwd_pkt_hdr_len_tot": "Fwd Header Length", "bwd_pkt_hdr_len_tot": "Bwd Header Length",
    "fwd_pkt_per_s": "Fwd Packets/s", "bwd_pkt_per_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length", "pkt_len_max": "Max Packet Length", "pkt_len_mean": "Packet Length Mean", "pkt_len_std": "Packet Length Std", "pkt_len_var": "Packet Length Variance",
    "flag_fin": "FIN Flag Count", "flag_SYN": "SYN Flag Count", "flag_rst": "RST Flag Count", "flag_psh": "PSH Flag Count", "flag_ack": "ACK Flag Count", "flag_urg": "URG Flag Count", "flag_cwr": "CWE Flag Count", "flag_ece": "ECE Flag Count", "down_up_ratio": "Down/Up Ratio",
    "fwd_bulk_bytes_mean": "Fwd Avg Bytes/Bulk", "fwd_bulk_pkt_mean": "Fwd Avg Packets/Bulk", "fwd_bulk_rate_mean": "Fwd Avg Bulk Rate",
    "bwd_bulk_bytes_mean": "Bwd Avg Bytes/Bulk", "bwd_bulk_pkt_mean": "Bwd Avg Packets/Bulk", "bwd_bulk_rate_mean": "Bwd Avg Bulk Rate",
    "fwd_subflow_pkt_mean": "Subflow Fwd Packets", "fwd_subflow_bytes_mean": "Subflow Fwd Bytes", "bwd_subflow_pkt_mean": "Subflow Bwd Packets", "bwd_subflow_bytes_mean": "Subflow Bwd Bytes",
    "fwd_tcp_init_win_bytes": "Init_Win_bytes_forward", "bwd_tcp_init_win_bytes": "Init_Win_bytes_backward",
    "fwd_non_empty_pkt_cnt": "act_data_pkt_fwd", "fwd_pkt_hdr_len_min": "min_seg_size_forward",
    "active_mean": "Active Mean", "active_std": "Active Std", "active_max": "Active Max", "active_min": "Active Min",
    "idle_mean": "Idle Mean", "idle_std": "Idle Std", "idle_max": "Idle Max", "idle_min": "Idle Min",
}

def get_actual_columns(path):
    if os.path.isfile(path):
        f = path
    else:
        f = sorted(glob.glob(os.path.join(path, "*.csv")))[0]
    peek = pd.read_csv(f, nrows=1, encoding="latin1")
    return [c.strip() for c in peek.columns if c.strip() not in ["Label", "label", " Label"]]

def main():
    print("="*60)
    print(" STAGE 1: FEATURE INVENTORY & MAPPING AUDIT ")
    print("="*60)
    
    inv_2018 = {v: k for k, v in IDS2018_TO_CICIDS2017.items()}
    inv_lycos = {v: k for k, v in LYCOS_TO_CICIDS2017.items()}
    
    actual_2017 = get_actual_columns(DATASETS["CICIDS2017"]) if os.path.exists(DATASETS["CICIDS2017"]) else CICIDS2017_FEATURES
    actual_2018 = get_actual_columns(DATASETS["CSE-CIC-IDS2018"]) if os.path.exists(DATASETS["CSE-CIC-IDS2018"]) else list(IDS2018_TO_CICIDS2017.keys())
    actual_lycos = get_actual_columns(DATASETS["Lycos-Unicas-IDS2018"]) if os.path.exists(DATASETS["Lycos-Unicas-IDS2018"]) else list(LYCOS_TO_CICIDS2017.keys())

    results = []
    
    # Track all expected CICIDS2017 features
    for feat in CICIDS2017_FEATURES:
        col_2017 = feat if feat in actual_2017 else "Missing/Derived"
        col_2018 = inv_2018.get(feat, "Missing")
        col_lycos = inv_lycos.get(feat, "Missing")
        
        # Check derived equivalents
        if feat == "Fwd Header Length.1":
            if "Fwd Header Len" in actual_2018: col_2018 = "Fwd Header Len (Derived)"
            if "fwd_pkt_hdr_len_tot" in actual_lycos: col_lycos = "fwd_pkt_hdr_len_tot (Derived)"
        if feat == "Avg Fwd Segment Size" and col_lycos == "Missing" and "fwd_pkt_len_mean" in actual_lycos: col_lycos = "fwd_pkt_len_mean (Derived)"
        if feat == "Avg Bwd Segment Size" and col_lycos == "Missing" and "bwd_pkt_len_mean" in actual_lycos: col_lycos = "bwd_pkt_len_mean (Derived)"
        if feat == "Average Packet Size" and col_lycos == "Missing" and "pkt_len_mean" in actual_lycos: col_lycos = "pkt_len_mean (Derived)"

        if col_2018 == "Missing" and col_lycos == "Missing":
            status = "Missing in Both External"
        elif col_2018 == "Missing":
            status = "Missing in 2018"
        elif col_lycos == "Missing":
            status = "Missing in Lycos"
        elif col_2017 == col_2018 == col_lycos:
            status = "Shared (Exact Name)"
        else:
            status = "Shared (Renamed/Derived)"

        results.append({
            "Feature": feat,
            "CIC2017": col_2017,
            "CIC2018": col_2018,
            "Lycos": col_lycos,
            "Status": status
        })
        
    # Check for extra features in external datasets
    mapped_2018_cols = set(inv_2018.values())
    for col in actual_2018:
        if col not in IDS2018_TO_CICIDS2017 and col not in ["Timestamp"]:
            results.append({
                "Feature": f"EXTRA: {col}",
                "CIC2017": "Missing",
                "CIC2018": col,
                "Lycos": "Missing",
                "Status": "Extra in 2018"
            })
            
    for col in actual_lycos:
        if col not in LYCOS_TO_CICIDS2017:
            results.append({
                "Feature": f"EXTRA: {col}",
                "CIC2017": "Missing",
                "CIC2018": "Missing",
                "Lycos": col,
                "Status": "Extra in Lycos"
            })

    df_inv = pd.DataFrame(results)
    out_csv = os.path.join(TABLES_DIR, "feature_inventory.csv")
    df_inv.to_csv(out_csv, index=False)
    print(f"\nFeature inventory saved to {out_csv}")
    
    # Print status summary
    status_counts = df_inv["Status"].value_counts()
    print("\nFeature Status Breakdown:")
    print(status_counts.to_string())
    
    # Generate Visualizations
    print("\nGenerating feature overlap bar charts ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=status_counts.values, y=status_counts.index, ax=ax, palette="cubehelix")
    ax.set_title("Stage 1: Cross-Dataset Feature Mismatch & Alignment Audit", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("Alignment Status", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    
    sns.despine()
    plt.tight_layout()
    
    out_img = os.path.join(FIGURES_DIR, "feature_overlap.png")
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved to {out_img}")
    print("\nDone!")

if __name__ == "__main__":
    main()
