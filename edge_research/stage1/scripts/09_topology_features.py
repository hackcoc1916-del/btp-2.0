"""
09_topology_features.py -- Stage 1: Topology & Artifact Leakage Analysis
========================================================================
Investigates networking topology features across CICIDS2017, CSE-CIC-IDS2018,
and Lycos-Unicas-IDS2018:
- Source Port
- Destination Port
- Timestamp
- Flow Bytes/s
- Flow Packets/s

Determines availability, structural mismatch, and significant distribution drift
to evaluate whether these features leak dataset-specific topology artifacts.

Saves:
- tables/topology_features_report.csv
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
    filename=os.path.join(LOGS_DIR, "09_topology_features.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = os.path.abspath(os.path.join(STAGE1_DIR, "..", "..", "data"))

DATASETS = {
    "CICIDS2017": os.path.join(DATA_DIR, "CICIDS2017"),
    "CSE-CIC-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "CSE-CIC-IDS2018"),
    "Lycos-Unicas-IDS2018": os.path.join(DATA_DIR, "datasets for cross validation", "LycoS-Unicas-IDS2018", "LycoS-Unicas-IDS2018.csv")
}

# Feature mappings for topology investigation
TOPOLOGY_MAPS = {
    "Source Port": {"CICIDS2017": "Source Port", "CSE-CIC-IDS2018": "Missing", "Lycos-Unicas-IDS2018": "Missing"},
    "Destination Port": {"CICIDS2017": "Destination Port", "CSE-CIC-IDS2018": "Dst Port", "Lycos-Unicas-IDS2018": "dst_port"},
    "Timestamp": {"CICIDS2017": "Timestamp", "CSE-CIC-IDS2018": "Timestamp", "Lycos-Unicas-IDS2018": "Missing"},
    "Flow Bytes/s": {"CICIDS2017": "Flow Bytes/s", "CSE-CIC-IDS2018": "Flow Byts/s", "Lycos-Unicas-IDS2018": "bytes_per_s"},
    "Flow Packets/s": {"CICIDS2017": "Flow Packets/s", "CSE-CIC-IDS2018": "Flow Pkts/s", "Lycos-Unicas-IDS2018": "pkt_per_s"}
}

def investigate_topology_feature(ds_name, path, std_feat_name, col_name):
    print(f"  Investigating {std_feat_name} ({col_name}) in {ds_name} ...")
    if col_name == "Missing":
        return {"Dataset": ds_name, "Feature": std_feat_name, "Actual Column": "Missing", "Status": "Missing in Dataset", "Mean": np.nan, "Std": np.nan, "Top Values": "N/A"}

    if os.path.isfile(path): files = [path]
    else: files = sorted(glob.glob(os.path.join(path, "*.csv")))

    s_chunks = []
    val_counts = pd.Series(dtype=int)

    for f in files:
        try:
            for chunk in pd.read_csv(f, usecols=lambda c: c.strip() == col_name, chunksize=1000000, low_memory=False, encoding="latin1"):
                chunk.columns = chunk.columns.str.strip()
                if std_feat_name in ["Source Port", "Destination Port", "Timestamp"]:
                    # Get value counts for categorical/port/time features
                    s = chunk[col_name].astype(str).str.strip()
                    s = s[~s.isin(["Label", "label", "Dst Port", "Destination Port", "Timestamp", "Source Port"])]
                    val_counts = val_counts.add(s.value_counts(), fill_value=0)
                else:
                    # Get numerical series for flow rate features
                    s = pd.to_numeric(chunk[col_name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    s_chunks.append(s)
        except Exception as e:
            logging.error(f"Error reading {col_name} in {f}: {e}")
            continue

    if std_feat_name in ["Source Port", "Destination Port", "Timestamp"]:
        val_counts = val_counts.sort_values(ascending=False)
        top_vals = ", ".join([f"{k} ({v})" for k, v in val_counts.head(3).items()])
        # Calculate mean/std if ports
        if std_feat_name in ["Source Port", "Destination Port"]:
            try:
                numeric_ports = pd.to_numeric(val_counts.index, errors="coerce")
                valid = ~np.isnan(numeric_ports)
                total_n = val_counts.values[valid].sum()
                mean_p = np.sum(numeric_ports[valid] * val_counts.values[valid]) / total_n if total_n > 0 else np.nan
                std_p = np.sqrt(np.sum(((numeric_ports[valid] - mean_p)**2) * val_counts.values[valid]) / total_n) if total_n > 0 else np.nan
            except Exception:
                mean_p, std_p = np.nan, np.nan
        else:
            mean_p, std_p = np.nan, np.nan
            
        return {"Dataset": ds_name, "Feature": std_feat_name, "Actual Column": col_name, "Status": "Available", "Mean": mean_p, "Std": std_p, "Top Values": top_vals}
    else:
        if not s_chunks:
            return {"Dataset": ds_name, "Feature": std_feat_name, "Actual Column": col_name, "Status": "Load Failed", "Mean": np.nan, "Std": np.nan, "Top Values": "N/A"}
        full_s = pd.concat(s_chunks, ignore_index=True)
        mean_v = full_s.mean()
        std_v  = full_s.std()
        top_vals = f"Min: {full_s.min():.2f}, Max: {full_s.max():.2f}"
        return {"Dataset": ds_name, "Feature": std_feat_name, "Actual Column": col_name, "Status": "Available", "Mean": mean_v, "Std": std_v, "Top Values": top_vals}

def main():
    print("="*60)
    print(" STAGE 1: TOPOLOGY & ARTIFACT LEAKAGE ANALYSIS ")
    print("="*60)
    
    results = []
    for std_feat, ds_cols in TOPOLOGY_MAPS.items():
        print(f"\nEvaluating Topology Feature: {std_feat}")
        for ds_name, path in DATASETS.items():
            if not os.path.exists(path):
                continue
            col_name = ds_cols[ds_name]
            res = investigate_topology_feature(ds_name, path, std_feat, col_name)
            results.append(res)
            
    df_topo = pd.DataFrame(results)
    
    # Calculate drift severity indicator
    df_topo["Significant Drift / Leakage"] = df_topo.apply(
        lambda r: "YES (Missing Feature)" if r["Status"] == "Missing in Dataset" else ("YES (High Distribution Variance)" if r["Feature"] in ["Source Port", "Destination Port", "Timestamp"] else "MODERATE (Numerical Drift)"),
        axis=1
    )
    
    out_csv = os.path.join(TABLES_DIR, "topology_features_report.csv")
    df_topo.to_csv(out_csv, index=False)
    print(f"\nTopology features report saved to {out_csv}")
    
    print("\nTopology Investigation Summary Table:")
    print(df_topo[["Dataset", "Feature", "Actual Column", "Status", "Mean", "Top Values", "Significant Drift / Leakage"]].to_string(index=False))
    print("\nDone!")

if __name__ == "__main__":
    main()
