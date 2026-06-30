import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

STAGE3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(STAGE3_DIR, "models")
LOGS_DIR = os.path.join(STAGE3_DIR, "logs")
STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE3_DIR, "../stage2/artifacts"))

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "06_lightgbm.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def train_lgb(task_type, parquet_file, scaler, encoder_file):
    data_path = os.path.join(STAGE2_ARTIFACTS, parquet_file)
    print(f"\nLoading {task_type} dataset from {data_path} ...")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_parquet(data_path)
    
    # Random subset for efficient training
    df_sample = df.sample(min(len(df), 100000), random_state=42)
    
    X = df_sample.drop(columns=["Label"])
    y_raw = np.array(df_sample["Label"].astype(str))
    
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    stage3_enc_path = os.path.join(MODELS_DIR, f"stage3_{task_type}_encoder.pkl")
    with open(stage3_enc_path, "wb") as f:
        pickle.dump(encoder, f)
    
    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else X.columns
    X_scaled = scaler.transform(X[scaler_cols])
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training LightGBM ({task_type}) on shape {X_train.shape} ...")
    
    # Attempt GPU training with fallback to CPU
    try:
        model = LGBMClassifier(n_estimators=100, device="gpu", random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train)
        print("  Successfully trained using GPU (device='gpu').")
    except Exception as e:
        logging.warning(f"LightGBM GPU fit failed: {e}. Fallback to CPU.")
        print("  LightGBM GPU fit failed or unavailable. Fallback to CPU.")
        model = LGBMClassifier(n_estimators=100, device="cpu", random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"  Validation Accuracy: {acc:.6f}")
    
    model_out = os.path.join(MODELS_DIR, f"lightgbm_{task_type}.txt")
    model.booster_.save_model(model_out)
    print(f"Saved model to {model_out}")
    
    # Save sklearn wrapper object as well for easy predict_proba in 08_model_comparison.py
    pkl_out = os.path.join(MODELS_DIR, f"lightgbm_{task_type}.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(model, f)
        
    logging.info(f"Successfully trained lightgbm_{task_type}. Validation Accuracy: {acc:.6f}")
    return model_out

def main():
    print("="*60)
    print(" STAGE 3: LIGHTGBM BASELINE TRAINING (GPU / FALLBACK) ")
    print("="*60)
    
    scaler_path = os.path.join(MODELS_DIR, "final_scaler.pkl")
    if not os.path.exists(scaler_path):
        logging.error(f"Final scaler not found at {scaler_path}. Run 02_scaler_comparison.py first.")
        sys.exit(1)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    out_multi = train_lgb("multiclass", "cic2017_multiclass.parquet", scaler, "label_encoder.pkl")
    out_bin = train_lgb("binary", "cic2017_binary.parquet", scaler, "binary_encoder.pkl")
    
    # Copy multiclass model to lightgbm.txt as requested by user
    general_out = os.path.join(MODELS_DIR, "lightgbm.txt")
    shutil.copy(out_multi, general_out)
    print(f"\nCopied primary multiclass model to {general_out}")
    
    logging.info("LightGBM baseline training complete.")
    print("\nDone!")

if __name__ == "__main__":
    main()
