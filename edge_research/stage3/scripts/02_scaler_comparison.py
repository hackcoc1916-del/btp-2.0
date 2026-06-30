import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

STAGE3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE3_DIR, "tables")
MODELS_DIR = os.path.join(STAGE3_DIR, "models")
LOGS_DIR = os.path.join(STAGE3_DIR, "logs")

STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE3_DIR, "../stage2/artifacts"))

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "02_scaler_comparison.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    print("="*60)
    print(" STAGE 3: SCALER COMPARISON & FINAL SELECTION ")
    print("="*60)
    
    # Load binary dataset sample for fast, robust scaler comparison
    data_path = os.path.join(STAGE2_ARTIFACTS, "cic2017_binary.parquet")
    print(f"Loading sample from {data_path} for scaler evaluation...")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        sys.exit(1)
        
    # Read a stratified subset of 100,000 rows to ensure fast execution
    df = pd.read_parquet(data_path)
    
    # Random sample of 100,000 rows
    df_sample = df.sample(min(len(df), 100000), random_state=42)
    
    X = df_sample.drop(columns=["Label"])
    y_str = df_sample["Label"].values
    y = np.array([0 if lbl == "BENIGN" else 1 for lbl in y_str])
    
    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scalers = {
        "StandardScaler": "standard_scaler.pkl",
        "RobustScaler": "robust_scaler.pkl",
        "QuantileTransformer": "quantile_scaler.pkl"
    }
    
    results = []
    best_scaler_name = None
    best_f1 = -1.0
    
    for name, pkl_file in scalers.items():
        pkl_path = os.path.join(STAGE2_ARTIFACTS, pkl_file)
        print(f"\nEvaluating Scaler: {name} ...")
        if not os.path.exists(pkl_path):
            logging.error(f"Scaler pkl not found at {pkl_path}")
            continue
            
        with open(pkl_path, "rb") as f:
            scaler = pickle.load(f)
            
        # Ensure we align feature names if scaler expects them
        scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else X_train.columns
        
        X_train_scaled = scaler.transform(X_train[scaler_cols])
        X_val_scaled = scaler.transform(X_val[scaler_cols])
        
        # Train baseline RandomForest classifier
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        
        print(f"  Accuracy: {acc:.6f}, F1-Score: {f1:.6f}, ROC-AUC: {auc:.6f}")
        
        results.append({
            "Scaler": name,
            "Accuracy": acc,
            "F1_Score": f1,
            "ROC_AUC": auc,
            "Artifact_Name": pkl_file
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_scaler_name = name
            
    df_results = pd.DataFrame(results)
    res_path = os.path.join(TABLES_DIR, "scaler_results.csv")
    df_results.to_csv(res_path, index=False)
    print(f"\nScaler comparison results saved to {res_path}")
    
    print("\nScaler Evaluation Summary Table:")
    print(df_results.to_string(index=False))
    
    # Save final scaler
    winning_file = scalers[best_scaler_name]
    src_scaler_path = os.path.join(STAGE2_ARTIFACTS, winning_file)
    dst_scaler_path = os.path.join(MODELS_DIR, "final_scaler.pkl")
    
    shutil.copy(src_scaler_path, dst_scaler_path)
    print(f"\nSelected {best_scaler_name} as the final scaler (Best F1: {best_f1:.6f}).")
    print(f"Copied to {dst_scaler_path}")
    
    logging.info(f"Scaler selection complete. Winning scaler: {best_scaler_name}")
    print("\nDone!")

if __name__ == "__main__":
    main()
