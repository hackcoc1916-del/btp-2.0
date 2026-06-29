import os
import sys
import logging
import pandas as pd

STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "06_feature_alignment.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    print("="*60)
    print(" STAGE 2: CROSS-DATASET FEATURE ALIGNMENT & ORDERING ")
    print("="*60)
    
    selected_feat_path = os.path.join(TABLES_DIR, "selected_features.csv")
    if not os.path.exists(selected_feat_path):
        logging.error(f"Selected features file not found at {selected_feat_path}. Run 01_feature_selection.py first.")
        sys.exit(1)
        
    df_selected = pd.read_csv(selected_feat_path)
    print(f"Loaded selected features list with {len(df_selected)} active features.")
    
    # Sort/fix the canonical order
    df_selected = df_selected.sort_values("Feature").reset_index(drop=True)
    df_selected["Feature_Index"] = df_selected.index
    
    # Establish canonical alignment mapping
    alignment_records = []
    for idx, row in df_selected.iterrows():
        alignment_records.append({
            "Feature_Index": idx,
            "Canonical_Feature": row["Feature"].strip(),
            "CIC2017_Raw": row["CIC2017"].strip(),
            "CIC2018_Raw": row["CIC2018"].strip(),
            "Lycos_Raw": row["Lycos"].strip(),
            "Alignment_Rule": "Direct Rename / Mapping",
            "Derivation_Strategy": "Exact Match / Zero-Fill if Omitted in Subset"
        })
        
    df_align = pd.DataFrame(alignment_records)
    
    out_path = os.path.join(TABLES_DIR, "feature_alignment.csv")
    df_align.to_csv(out_path, index=False)
    
    print(f"\nFeature alignment specification saved to {out_path}")
    print(f"  Total Aligned Features: {len(df_align)}")
    print(f"  Ordering Guaranteed: Identical across CICIDS2017, CSE-CIC-IDS2018, and Lycos-Unicas-IDS2018")
    
    print("\nFeature Alignment Sample (Top 15 Features):")
    print(df_align.head(15)[["Feature_Index", "Canonical_Feature", "CIC2017_Raw", "CIC2018_Raw", "Lycos_Raw"]].to_string(index=False))
    
    logging.info(f"Feature alignment complete. Aligned {len(df_align)} features with identical ordering.")
    print("\nDone!")

if __name__ == "__main__":
    main()
