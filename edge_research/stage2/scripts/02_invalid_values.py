import os
import sys
import logging
import pandas as pd

STAGE1_TABLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../stage1/tables"))
STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "02_invalid_values.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    print("="*60)
    print(" STAGE 2: INVALID VALUES AUDIT & REPLACEMENT STRATEGY ")
    print("="*60)
    
    missing_tbl_path = os.path.join(STAGE1_TABLES_DIR, "missing_values.csv")
    if not os.path.exists(missing_tbl_path):
        logging.error(f"Stage 1 missing values table not found at {missing_tbl_path}. Cannot proceed.")
        sys.exit(1)
        
    df_missing = pd.read_csv(missing_tbl_path)
    print(f"Loaded Stage 1 invalid value statistics across {len(df_missing)} dataset-feature combinations.")
    
    # Establish the formal transformation rules
    df_missing["Inf_Replacement_Rule"] = "Replace with NaN"
    df_missing["NegInf_Replacement_Rule"] = "Replace with NaN"
    df_missing["Total_Post_Replacement_NaN"] = df_missing["NaN Count"] + df_missing["Inf Count"] + df_missing["-Inf Count"]
    
    out_path = os.path.join(TABLES_DIR, "invalid_values.csv")
    df_missing.to_csv(out_path, index=False)
    
    print(f"\nInvalid values statistics and replacement mapping saved to {out_path}")
    
    summary = df_missing.groupby("Dataset")[["NaN Count", "Inf Count", "-Inf Count", "Total_Post_Replacement_NaN"]].sum().reset_index()
    print("\nDataset-level Invalid Values Summary:")
    print(summary.to_string(index=False))
    
    top_invalid = df_missing[df_missing["Total_Post_Replacement_NaN"] > 0].sort_values("Total_Post_Replacement_NaN", ascending=False).head(15)
    print("\nTop 15 Features with Highest Invalid Value Counts:")
    print(top_invalid[["Dataset", "Feature", "NaN Count", "Inf Count", "Total_Post_Replacement_NaN"]].to_string(index=False))
    
    logging.info(f"Invalid values audit complete. Total invalid values across all datasets: {summary['Total_Post_Replacement_NaN'].sum()}")
    print("\nDone!")

if __name__ == "__main__":
    main()
