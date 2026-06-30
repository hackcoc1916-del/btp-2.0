import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

STAGE3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE3_DIR, "tables")
MODELS_DIR = os.path.join(STAGE3_DIR, "models")
FIGURES_DIR = os.path.join(STAGE3_DIR, "figures")
LOGS_DIR = os.path.join(STAGE3_DIR, "logs")
STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE3_DIR, "../stage2/artifacts"))

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "08_model_comparison.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def evaluate_models(task_type, parquet_file, scaler, encoder_file):
    data_path = os.path.join(STAGE2_ARTIFACTS, parquet_file)
    print(f"\nEvaluating {task_type} models on validation split from {data_path} ...")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_parquet(data_path)
    
    # Use the same random validation subset as during training
    df_sample = df.sample(min(len(df), 100000), random_state=42)
    
    X = df_sample.drop(columns=["Label"])
    y_str = np.array(df_sample["Label"].astype(str))
    
    enc_path = os.path.join(MODELS_DIR, f"stage3_{task_type}_encoder.pkl")
    with open(enc_path, "rb") as f:
        encoder = pickle.load(f)
    y_encoded = encoder.transform(y_str)
    
    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else X.columns
    X_scaled = scaler.transform(X[scaler_cols])
    
    # Split to get validation set
    _, X_val, _, y_val_str = train_test_split(X_scaled, y_str, test_size=0.2, random_state=42, stratify=y_str)
    _, _, _, y_val_encoded = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    models = {
        "Logistic Regression": f"logistic_{task_type}.pkl",
        "Random Forest": f"random_forest_{task_type}.pkl",
        "XGBoost": f"xgboost_{task_type}.pkl",
        "LightGBM": f"lightgbm_{task_type}.pkl",
        "Extra Trees": f"extra_trees_{task_type}.pkl"
    }
    
    results = []
    
    for name, pkl_file in models.items():
        pkl_path = os.path.join(MODELS_DIR, pkl_file)
        print(f"  Evaluating {name} ...")
        if not os.path.exists(pkl_path):
            logging.error(f"Model pkl not found at {pkl_path}")
            continue
            
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
            
        is_encoded_model = name in ["XGBoost", "LightGBM"]
        y_true = y_val_encoded if is_encoded_model else y_val_str
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        if task_type == "binary":
            # Determine pos_label
            pos_label = 1 if is_encoded_model else "ATTACK"
            prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob is not None else 0.0
        else:
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") if y_prob is not None else 0.0
            
        results.append({
            "Task_Type": task_type,
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": auc,
            "MCC": mcc
        })
        
    return pd.DataFrame(results)

def main():
    print("="*60)
    print(" STAGE 3: COMPREHENSIVE BASELINE MODEL COMPARISON ")
    print("="*60)
    
    scaler_path = os.path.join(MODELS_DIR, "final_scaler.pkl")
    if not os.path.exists(scaler_path):
        logging.error(f"Final scaler not found at {scaler_path}. Run 02_scaler_comparison.py first.")
        sys.exit(1)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    df_multi = evaluate_models("multiclass", "cic2017_multiclass.parquet", scaler, "label_encoder.pkl")
    df_bin = evaluate_models("binary", "cic2017_binary.parquet", scaler, "binary_encoder.pkl")
    
    df_results = pd.concat([df_multi, df_bin], ignore_index=True)
    
    out_path = os.path.join(TABLES_DIR, "model_comparison.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nModel comparison results saved to {out_path}")
    print("\nModel Comparison Table:")
    print(df_results.to_string(index=False))
    
    # Generate bar plots for F1-Score and ROC-AUC
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_results, x="Model", y="F1_Score", hue="Task_Type", palette="viridis")
    plt.title("Stage 3 Baseline Model Comparison: F1-Score (Binary vs Multiclass)")
    plt.ylabel("F1-Score")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig_f1 = os.path.join(FIGURES_DIR, "model_comparison_f1.png")
    plt.savefig(fig_f1, dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_results, x="Model", y="ROC_AUC", hue="Task_Type", palette="magma")
    plt.title("Stage 3 Baseline Model Comparison: ROC-AUC (Binary vs Multiclass)")
    plt.ylabel("ROC-AUC")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig_auc = os.path.join(FIGURES_DIR, "model_comparison_roc_auc.png")
    plt.savefig(fig_auc, dpi=300)
    plt.close()
    
    print(f"Saved comparison figures to {FIGURES_DIR}")
    logging.info("Model comparison evaluation complete.")
    print("\nDone!")

if __name__ == "__main__":
    main()
