import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
ARTIFACTS_DIR = os.path.join(STAGE2_DIR, "artifacts")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")

DATASETS = {
    "cic2017": os.path.abspath(os.path.join(STAGE2_DIR, "../../data/CICIDS2017")),
    "cic2018": os.path.abspath(os.path.join(STAGE2_DIR, "../../data/datasets for cross validation/CSE-CIC-IDS2018")),
    "lycos": os.path.abspath(os.path.join(STAGE2_DIR, "../../data/datasets for cross validation/LycoS-Unicas-IDS2018"))
}

for d in [TABLES_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "09_dataset_export.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Canonical Attack Mapping
ATTACK_MAPPING = {
    "BENIGN": "BENIGN", "Benign": "BENIGN",
    "DoS Hulk": "DOS_DDOS", "DoS GoldenEye": "DOS_DDOS", "DoS slowloris": "DOS_DDOS", "DoS Slowloris": "DOS_DDOS", "DoS Slowhttptest": "DOS_DDOS",
    "DDoS": "DOS_DDOS", "DDOS attack-HOIC": "DOS_DDOS", "DDoS attacks-LOIC-HTTP": "DOS_DDOS",
    "DoS attacks-Hulk": "DOS_DDOS", "DoS attacks-SlowHTTPTest": "DOS_DDOS", "DoS attacks-GoldenEye": "DOS_DDOS",
    "DoS attacks-Slowloris": "DOS_DDOS", "DDOS attack-LOIC-UDP": "DOS_DDOS",
    "DDoS HOIC": "DOS_DDOS", "DDoS LOIC-HTTP": "DOS_DDOS", "DDoS LOIC-UDP": "DOS_DDOS",
    "PortScan": "PROBING",
    "FTP-Patator": "BRUTE_FORCE", "SSH-Patator": "BRUTE_FORCE", "FTP-BruteForce": "BRUTE_FORCE", "SSH-Bruteforce": "BRUTE_FORCE",
    "Web Attack – Brute Force": "WEB_ATTACK", "Web Attack – XSS": "WEB_ATTACK", "Web Attack – Sql Injection": "WEB_ATTACK",
    "Brute Force -Web": "WEB_ATTACK", "Brute Force -XSS": "WEB_ATTACK", "SQL Injection": "WEB_ATTACK",
    "Web Attack - Brute Force": "WEB_ATTACK", "Web Attack - XSS": "WEB_ATTACK", "Web Attack - Sql Injection": "WEB_ATTACK",
    "Bot": "BOT",
    "Infiltration": "REMOVE", "Infilteration": "REMOVE", "Heartbleed": "REMOVE"
}

def export_dataset_streaming(ds_key, data_dir, df_align, median_imputer):
    print(f"\nExporting dataset: {ds_key} ...")
    multiclass_out = os.path.join(ARTIFACTS_DIR, f"{ds_key}_multiclass.parquet")
    binary_out = os.path.join(ARTIFACTS_DIR, f"{ds_key}_binary.parquet")
    
    # Establish mapping for this dataset
    features = df_align["Canonical_Feature"].tolist()
    raw_col_name = "CIC2017_Raw" if ds_key == "cic2017" else ("CIC2018_Raw" if ds_key == "cic2018" else "Lycos_Raw")
    raw_cols = df_align[raw_col_name].tolist()
    feat_map = dict(zip(raw_cols, features))
    raw_cols_clean = {c.strip() for c in raw_cols}
    feat_map_clean = {k.strip(): v for k, v in feat_map.items()}
    
    writer_multi = None
    writer_bin = None
    total_rows = 0
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    for f in files:
        path = os.path.join(data_dir, f)
        print(f"  Streaming chunks from {f} ...")
        try:
            for chunk in pd.read_csv(path, chunksize=100000, encoding="latin1", usecols=lambda c: c.strip() in raw_cols_clean or c.strip() in ["Label", "label"]):
                chunk.columns = [c.strip() for c in chunk.columns]
                
                lbl_col = None
                for c in ["Label", "label"]:
                    if c in chunk.columns: lbl_col = c; break
                if lbl_col is None: lbl_col = chunk.columns[-1]
                
                chunk = chunk.rename(columns=lambda c: feat_map_clean.get(c, c))
                
                # Standardize Multiclass Target
                raw_lbls = chunk[lbl_col].astype(str).str.strip().str.replace("\u2013", "-").str.replace("\x96", "-")
                chunk["Target_Multiclass"] = raw_lbls.map(ATTACK_MAPPING).fillna("OTHER")
                
                # Drop REMOVE classes
                chunk = chunk[chunk["Target_Multiclass"] != "REMOVE"]
                if chunk.empty: continue
                
                # Derive Binary Target
                chunk["Target_Binary"] = chunk["Target_Multiclass"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")
                
                # Align features
                for ft in features:
                    if ft not in chunk.columns: chunk[ft] = 0.0
                    
                df_feats = chunk[features].copy()
                for c in features:
                    df_feats[c] = pd.to_numeric(df_feats[c], errors="coerce")
                df_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Impute missing values using pre-fitted median imputer
                imputer_cols = list(median_imputer.feature_names_in_)
                X_imp = median_imputer.transform(df_feats[imputer_cols])
                df_clean = pd.DataFrame(X_imp, columns=imputer_cols)
                
                # Reorder back to canonical sorted features list
                df_clean = df_clean[features]
                
                # Cast to float32 to save disk space and memory
                for c in features:
                    df_clean[c] = df_clean[c].astype(np.float32)
                    
                df_multi = df_clean.copy()
                df_multi["Label"] = chunk["Target_Multiclass"].values
                
                df_bin = df_clean.copy()
                df_bin["Label"] = chunk["Target_Binary"].values
                
                # Write Parquet chunks
                table_multi = pa.Table.from_pandas(df_multi)
                table_bin = pa.Table.from_pandas(df_bin)
                
                if writer_multi is None:
                    writer_multi = pq.ParquetWriter(multiclass_out, table_multi.schema, compression="SNAPPY")
                if writer_bin is None:
                    writer_bin = pq.ParquetWriter(binary_out, table_bin.schema, compression="SNAPPY")
                    
                writer_multi.write_table(table_multi)
                writer_bin.write_table(table_bin)
                total_rows += len(df_clean)
        except Exception as e:
            logging.error(f"Error streaming {f}: {e}")
            continue
            
    if writer_multi: writer_multi.close()
    if writer_bin: writer_bin.close()
    
    print(f"  Successfully exported {total_rows} clean rows to {multiclass_out} and {binary_out}")
    return total_rows

def main():
    print("="*60)
    print(" STAGE 2: MASSIVE PARQUET DATASET EXPORT & SERIALIZATION ")
    print("="*60)
    
    align_path = os.path.join(TABLES_DIR, "feature_alignment.csv")
    if not os.path.exists(align_path):
        logging.error(f"Feature alignment file not found at {align_path}. Run 06_feature_alignment.py first.")
        sys.exit(1)
        
    df_align = pd.read_csv(align_path)
    features = df_align["Canonical_Feature"].tolist()
    
    # Save feature_columns.pkl
    feat_cols_path = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")
    with open(feat_cols_path, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved feature_columns.pkl with {len(features)} ordered features.")
    
    # Fit and save Label Encoders
    multiclass_labels = ["BENIGN", "DOS_DDOS", "PROBING", "BRUTE_FORCE", "WEB_ATTACK", "BOT", "OTHER"]
    binary_labels = ["BENIGN", "ATTACK"]
    
    label_encoder = LabelEncoder()
    label_encoder.fit(multiclass_labels)
    le_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label_encoder.pkl with classes: {label_encoder.classes_}")
    
    binary_encoder = LabelEncoder()
    binary_encoder.fit(binary_labels)
    be_path = os.path.join(ARTIFACTS_DIR, "binary_encoder.pkl")
    with open(be_path, "wb") as f:
        pickle.dump(binary_encoder, f)
    print(f"Saved binary_encoder.pkl with classes: {binary_encoder.classes_}")
    
    # Load median imputer for data cleaning during export
    med_path = os.path.join(ARTIFACTS_DIR, "median_imputer.pkl")
    if not os.path.exists(med_path):
        logging.error(f"Median imputer not found at {med_path}. Run 03_missing_values.py first.")
        sys.exit(1)
    with open(med_path, "rb") as f:
        median_imputer = pickle.load(f)
        
    # Export datasets
    for ds_key, data_dir in DATASETS.items():
        if os.path.exists(data_dir):
            export_dataset_streaming(ds_key, data_dir, df_align, median_imputer)
        else:
            logging.warning(f"Data directory {data_dir} not found. Skipping {ds_key}.")
            
    logging.info("Dataset export complete.")
    print("\nDone!")

if __name__ == "__main__":
    main()
