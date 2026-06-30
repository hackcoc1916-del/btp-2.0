import os
import sys
import logging
import pandas as pd
import numpy as np

# Establish directory structure
STAGE3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE3_DIR, "tables")
MODELS_DIR = os.path.join(STAGE3_DIR, "models")
REPORTS_DIR = os.path.join(STAGE3_DIR, "reports")
FIGURES_DIR = os.path.join(STAGE3_DIR, "figures")
LOGS_DIR = os.path.join(STAGE3_DIR, "logs")

for d in [TABLES_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "01_dataset_loading.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE3_DIR, "../stage2/artifacts"))

def verify_dataset(file_name, task_type):
    file_path = os.path.join(STAGE2_ARTIFACTS, file_name)
    logging.info(f"Loading {task_type} dataset from {file_path}...")
    print(f"\nLoading {task_type} dataset: {file_name} ...")
    
    if not os.path.exists(file_path):
        logging.error(f"Dataset file not found at {file_path}")
        sys.exit(1)
        
    df = pd.read_parquet(file_path)
    
    total_rows, total_cols = df.shape
    feature_cols = [c for c in df.columns if c != "Label"]
    
    print(f"  Total Rows: {total_rows}")
    print(f"  Total Columns: {total_cols} ({len(feature_cols)} features + 1 Label)")
    
    # Check missing values
    missing_count = df.isna().sum().sum()
    print(f"  Total Missing Values (NaN): {missing_count}")
    
    # Check infinite values
    inf_count = np.isinf(df[feature_cols].values).sum()
    print(f"  Total Infinite Values (inf): {inf_count}")
    
    # Label distribution
    lbl_dist = df["Label"].value_counts().reset_index()
    lbl_dist.columns = ["Label", "Frequency"]
    lbl_dist["Percentage"] = (lbl_dist["Frequency"] / total_rows) * 100
    
    print(f"\n  {task_type.capitalize()} Label Distribution:")
    print(lbl_dist.to_string(index=False))
    
    stats_list = []
    for _, r in lbl_dist.iterrows():
        stats_list.append({
            "Dataset": file_name,
            "Task_Type": task_type,
            "Total_Rows": total_rows,
            "Feature_Count": len(feature_cols),
            "Missing_Values": missing_count,
            "Infinite_Values": inf_count,
            "Label_Class": r["Label"],
            "Frequency": r["Frequency"],
            "Percentage": r["Percentage"]
        })
        
    logging.info(f"Successfully verified {file_name}. Rows: {total_rows}, Features: {len(feature_cols)}")
    return pd.DataFrame(stats_list)

def main():
    print("="*60)
    print(" STAGE 3: BASELINE DATASET LOADING & AUDIT ")
    print("="*60)
    
    df_multi = verify_dataset("cic2017_multiclass.parquet", "multiclass")
    df_bin = verify_dataset("cic2017_binary.parquet", "binary")
    
    df_stats = pd.concat([df_multi, df_bin], ignore_index=True)
    
    stats_path = os.path.join(TABLES_DIR, "dataset_statistics.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"\nDataset statistics saved to {stats_path}")
    
    logging.info("Stage 3 dataset loading audit complete.")
    print("\nDone!")

if __name__ == "__main__":
    main()
