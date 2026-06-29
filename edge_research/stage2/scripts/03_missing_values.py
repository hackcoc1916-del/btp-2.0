import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

STAGE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE2_DIR, "tables")
ARTIFACTS_DIR = os.path.join(STAGE2_DIR, "artifacts")
LOGS_DIR = os.path.join(STAGE2_DIR, "logs")
DATA_DIR = os.path.abspath(os.path.join(STAGE2_DIR, "../../data/CICIDS2017"))

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "03_missing_values.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    print("="*60)
    print(" STAGE 2: IMPUTATION STRATEGY EVALUATION ")
    print("="*60)
    
    selected_feat_path = os.path.join(TABLES_DIR, "selected_features.csv")
    if not os.path.exists(selected_feat_path):
        logging.error(f"Selected features file not found at {selected_feat_path}. Run 01_feature_selection.py first.")
        sys.exit(1)
        
    df_feats = pd.read_csv(selected_feat_path)
    # We include SELECTED and TOPOLOGY_CANDIDATE features for fitting the imputers
    features = df_feats["Feature"].tolist()
    cic_cols = df_feats["CIC2017"].tolist()
    feat_map = dict(zip(cic_cols, features))
    
    print(f"Loading representative sample from CICIDS2017 to evaluate imputation strategies across {len(features)} features...")
    
    # Load sample from CICIDS2017 to fit imputers and evaluate variance preservation
    sample_dfs = []
    total_loaded = 0
    for f in os.listdir(DATA_DIR):
        if f.endswith(".csv"):
            path = os.path.join(DATA_DIR, f)
            try:
                chunk = pd.read_csv(path, nrows=50000, encoding="latin1", usecols=lambda c: c.strip() in cic_cols)
                chunk = chunk.rename(columns=lambda c: feat_map.get(c.strip(), c.strip()))
                sample_dfs.append(chunk)
                total_loaded += len(chunk)
                if total_loaded >= 200000:
                    break
            except Exception as e:
                logging.error(f"Error reading {f}: {e}")
                continue
                
    if not sample_dfs:
        logging.error("Failed to load sample data for imputation evaluation.")
        sys.exit(1)
        
    df_sample = pd.concat(sample_dfs, ignore_index=True)
    
    # Ensure all features exist
    for ft in features:
        if ft not in df_sample.columns:
            df_sample[ft] = 0.0
            
    df_sample = df_sample[features].copy()
    
    # Convert to numeric and replace inf/-inf with NaN
    for col in features:
        df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce")
    df_sample.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check current missingness, if too low, simulate missingness on top features known to have missing values in full dataset
    simulated = False
    if df_sample.isna().sum().sum() < 1000:
        simulated = True
        np.random.seed(42)
        # Flow Bytes/s and Flow Packets/s are known to have missing/inf values
        for c in ["Flow Bytes/s", "Flow Packets/s"]:
            if c in df_sample.columns:
                mask = np.random.rand(len(df_sample)) < 0.05
                df_sample.loc[mask, c] = np.nan
                
    print(f"Sample data ready. Total rows: {len(df_sample)}, Total missing/invalid values: {df_sample.isna().sum().sum()}")
    
    # 1. Fit and Evaluate Median Imputer
    median_imputer = SimpleImputer(strategy="median")
    df_median = pd.DataFrame(median_imputer.fit_transform(df_sample), columns=features)
    
    # 2. Fit and Evaluate Zero Imputer
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    df_zero = pd.DataFrame(zero_imputer.fit_transform(df_sample), columns=features)
    
    # Evaluate Variance Preservation & Distribution Preservation
    eval_records = []
    for col in features:
        orig_var = df_sample[col].var()
        if pd.isna(orig_var) or orig_var == 0:
            continue
        med_var = df_median[col].var()
        zero_var = df_zero[col].var()
        
        orig_mean = df_sample[col].mean()
        med_mean = df_median[col].mean()
        zero_mean = df_zero[col].mean()
        
        # Variance preservation ratio (closer to 1.0 is better)
        med_var_ratio = med_var / orig_var
        zero_var_ratio = zero_var / orig_var
        
        eval_records.append({
            "Feature": col,
            "Orig_Mean": orig_mean,
            "Median_Mean": med_mean,
            "Zero_Mean": zero_mean,
            "Orig_Var": orig_var,
            "Median_Var_Ratio": med_var_ratio,
            "Zero_Var_Ratio": zero_var_ratio
        })
        
    df_eval = pd.DataFrame(eval_records)
    eval_out = os.path.join(TABLES_DIR, "imputation_evaluation.csv")
    df_eval.to_csv(eval_out, index=False)
    
    print(f"\nImputation evaluation report saved to {eval_out}")
    print("\nImputation Comparison on Top Features (Variance Preservation Ratio closer to 1.0 is superior):")
    # Show features with largest differences or known missingness
    top_diff = df_eval.iloc[(df_eval["Median_Var_Ratio"] - df_eval["Zero_Var_Ratio"]).abs().argsort()[::-1]].head(10)
    print(top_diff[["Feature", "Orig_Mean", "Median_Mean", "Zero_Mean", "Median_Var_Ratio", "Zero_Var_Ratio"]].to_string(index=False))
    
    # Save both imputers without selecting the final strategy yet
    med_path = os.path.join(ARTIFACTS_DIR, "median_imputer.pkl")
    zero_path = os.path.join(ARTIFACTS_DIR, "zero_imputer.pkl")
    
    with open(med_path, "wb") as f:
        pickle.dump(median_imputer, f)
    with open(zero_path, "wb") as f:
        pickle.dump(zero_imputer, f)
        
    print(f"\nSaved Median Imputer artifact to: {med_path}")
    print(f"Saved Zero Imputer artifact to:   {zero_path}")
    print("\nNOTE: Final strategy selection deferred as per Stage 2 requirements.")
    
    logging.info(f"Imputation evaluation complete. Artifacts saved to {ARTIFACTS_DIR}.")
    print("\nDone!")

if __name__ == "__main__":
    main()
