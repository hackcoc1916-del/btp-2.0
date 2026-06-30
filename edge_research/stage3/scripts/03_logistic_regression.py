import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

STAGE3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(STAGE3_DIR, "models")
LOGS_DIR = os.path.join(STAGE3_DIR, "logs")
STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE3_DIR, "../stage2/artifacts"))

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "03_logistic_regression.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def train_logistic(task_type, parquet_file, scaler):
    data_path = os.path.join(STAGE2_ARTIFACTS, parquet_file)
    print(f"\nLoading {task_type} dataset from {data_path} ...")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_parquet(data_path)
    
    # Use a robust random subset of 100,000 rows to ensure fast convergence and avoid OOM
    df_sample = df.sample(min(len(df), 100000), random_state=42)
    
    X = df_sample.drop(columns=["Label"])
    y = np.array(df_sample["Label"].astype(str))
    
    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else X.columns
    X_scaled = scaler.transform(X[scaler_cols])
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Logistic Regression ({task_type}) on shape {X_train.shape} ...")
    model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"  Validation Accuracy: {acc:.6f}")
    
    model_out = os.path.join(MODELS_DIR, f"logistic_{task_type}.pkl")
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_out}")
    
    logging.info(f"Successfully trained logistic_{task_type}. Validation Accuracy: {acc:.6f}")
    return model

def main():
    print("="*60)
    print(" STAGE 3: LOGISTIC REGRESSION BASELINE TRAINING ")
    print("="*60)
    
    scaler_path = os.path.join(MODELS_DIR, "final_scaler.pkl")
    if not os.path.exists(scaler_path):
        logging.error(f"Final scaler not found at {scaler_path}. Run 02_scaler_comparison.py first.")
        sys.exit(1)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    model_multi = train_logistic("multiclass", "cic2017_multiclass.parquet", scaler)
    model_bin = train_logistic("binary", "cic2017_binary.parquet", scaler)
    
    # Save combined dictionary to logistic.pkl as requested by user
    combined_out = os.path.join(MODELS_DIR, "logistic.pkl")
    with open(combined_out, "wb") as f:
        pickle.dump({"multiclass": model_multi, "binary": model_bin}, f)
    print(f"\nSaved combined models to {combined_out}")
    
    logging.info("Logistic regression baseline training complete.")
    print("\nDone!")

if __name__ == "__main__":
    main()
