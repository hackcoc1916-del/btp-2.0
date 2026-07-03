"""
Stage 4 — Script 03: Binary Transfer Evaluation (BENIGN vs ATTACK)
===================================================================
Train:  CICIDS2017 binary models (Stage 3)
Test:   CIC2018 binary + Lycos binary

Evaluates all 5 baseline binary models on both external datasets.
"""

import os
import sys
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Directory structure ──────────────────────────────────────────────
STAGE4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE4_DIR, "tables")
FIGURES_DIR = os.path.join(STAGE4_DIR, "figures")
LOGS_DIR = os.path.join(STAGE4_DIR, "logs")

STAGE3_MODELS = os.path.abspath(os.path.join(STAGE4_DIR, "../stage3/models"))
STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE4_DIR, "../stage2/artifacts"))

for d in [TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "03_binary_transfer.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

SAMPLE_SIZE = 500_000

# ── Model registry (binary) ─────────────────────────────────────────
MODELS = {
    "Logistic Regression": {"file": "logistic_binary.pkl",      "encoded": False},
    "Random Forest":       {"file": "random_forest_binary.pkl", "encoded": False},
    "Extra Trees":         {"file": "extra_trees_binary.pkl",   "encoded": False},
    "XGBoost":             {"file": "xgboost_binary.pkl",       "encoded": True},
    "LightGBM":            {"file": "lightgbm_binary.pkl",      "encoded": True},
}

DATASETS = {
    "CIC2018": "cic2018_binary.parquet",
    "Lycos":   "lycos_binary.parquet",
}


def load_artifacts():
    """Load scaler and binary encoder from Stage 3."""
    scaler_path = os.path.join(STAGE3_MODELS, "final_scaler.pkl")
    encoder_path = os.path.join(STAGE3_MODELS, "stage3_binary_encoder.pkl")

    for p, name in [(scaler_path, "scaler"), (encoder_path, "encoder")]:
        if not os.path.exists(p):
            logging.error(f"{name} not found at {p}")
            sys.exit(1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return scaler, encoder


def evaluate_on_dataset(dataset_name, parquet_file, scaler, encoder):
    """Evaluate all binary models on a single external dataset."""
    data_path = os.path.join(STAGE2_ARTIFACTS, parquet_file)
    print(f"\n{'-'*50}")
    print(f"Loading {dataset_name} binary dataset from {data_path} ...")

    if not os.path.exists(data_path):
        logging.error(f"Dataset not found: {data_path}")
        return []

    df = pd.read_parquet(data_path)
    print(f"  Total rows: {len(df):,}")
    print(f"  Labels present: {df['Label'].unique().tolist()}")

    # Filter to known binary labels
    known_classes = set(encoder.classes_)
    original_len = len(df)
    df = df[df["Label"].isin(known_classes)].copy()
    filtered_out = original_len - len(df)
    if filtered_out > 0:
        print(f"  Filtered out {filtered_out:,} rows with unknown labels")

    # Stratified sample
    if len(df) > SAMPLE_SIZE:
        df_sampled = []
        for lbl, group in df.groupby("Label"):
            n_samples = min(len(group), max(100, int(SAMPLE_SIZE * len(group) / original_len)))
            df_sampled.append(group.sample(n_samples, random_state=42))
        df = pd.concat(df_sampled, ignore_index=True)
        print(f"  Stratified sample: {len(df):,} rows")

    X = df.drop(columns=["Label"])
    y_str = np.array(df["Label"].astype(str))

    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else X.columns.tolist()
    missing = set(scaler_cols) - set(X.columns)
    if missing:
        logging.error(f"Missing features in {dataset_name}: {missing}")
        return []

    X_scaled = scaler.transform(X[scaler_cols])
    print(f"  Features: {X_scaled.shape[1]}, Labels: {np.unique(y_str).tolist()}")

    results = []
    for name, meta in MODELS.items():
        pkl_path = os.path.join(STAGE3_MODELS, meta["file"])
        if not os.path.exists(pkl_path):
            logging.warning(f"Model not found: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        # Prepare labels
        if meta["encoded"]:
            y_true = encoder.transform(y_str)
            pos_label = 1
        else:
            y_true = y_str
            pos_label = "ATTACK"

        X_eval = X_scaled

        print(f"\n  Evaluating {name} on {len(X_eval):,} samples ...")

        t_start = time.time()
        y_pred = model.predict(X_eval)
        pred_time = time.time() - t_start

        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_eval)
            except Exception:
                y_prob = None

        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

        auroc = 0.0
        if y_prob is not None:
            try:
                auroc = roc_auc_score(y_true, y_prob[:, 1])
            except Exception:
                auroc = 0.0

        print(f"    Accuracy: {acc:.6f}, F1: {f1:.6f}, MCC: {mcc:.6f}, AUROC: {auroc:.6f}")

        # Confusion matrix
        all_labels = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
        if meta["encoded"]:
            cm_labels = [encoder.inverse_transform([l])[0] for l in all_labels]
        else:
            cm_labels = [str(l) for l in all_labels]

        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=cm_labels, yticklabels=cm_labels)
        plt.title(f"{dataset_name} Binary — {name}")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"cm_{dataset_name.lower()}_binary_{name.lower().replace(' ', '_')}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        results.append({
            "Dataset": dataset_name,
            "Task_Type": "binary",
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": auroc,
            "MCC": mcc,
            "Prediction_Time_sec": pred_time,
            "Samples_Evaluated": len(X_eval)
        })

        logging.info(f"{dataset_name}/{name}: Acc={acc:.6f}, F1={f1:.6f}, MCC={mcc:.6f}")

    return results


def main():
    print("=" * 60)
    print(" STAGE 4: BINARY TRANSFER EVALUATION (BENIGN vs ATTACK) ")
    print("=" * 60)

    scaler, encoder = load_artifacts()

    all_results = []
    for ds_name, ds_file in DATASETS.items():
        rows = evaluate_on_dataset(ds_name, ds_file, scaler, encoder)
        all_results.extend(rows)

    df_results = pd.DataFrame(all_results)
    out_path = os.path.join(TABLES_DIR, "binary_transfer_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\n" + df_results.to_string(index=False))

    logging.info("Script 03 complete — binary transfer evaluation finished.")
    print("\nDone!")


if __name__ == "__main__":
    main()
