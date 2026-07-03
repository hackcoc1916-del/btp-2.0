"""
Stage 4 — Script 02: Lycos-Unicas-IDS2018 Multiclass Generalization Evaluation
================================================================================
Train:  CICIDS2017 (Stage 3 models)
Test:   Lycos-Unicas-IDS2018

Evaluates all 5 baseline models on the completely unseen Lycos
multiclass dataset using the identical scaler and encoder from Stage 3.
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
REPORTS_DIR = os.path.join(STAGE4_DIR, "reports")
LOGS_DIR = os.path.join(STAGE4_DIR, "logs")

STAGE3_MODELS = os.path.abspath(os.path.join(STAGE4_DIR, "../stage3/models"))
STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE4_DIR, "../stage2/artifacts"))

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "02_lycos_multiclass.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

SAMPLE_SIZE = 500_000

# ── Model registry ──────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": {"file": "logistic_multiclass.pkl",   "encoded": False},
    "Random Forest":       {"file": "random_forest_multiclass.pkl", "encoded": False},
    "Extra Trees":         {"file": "extra_trees_multiclass.pkl",   "encoded": False},
    "XGBoost":             {"file": "xgboost_multiclass.pkl",       "encoded": True},
    "LightGBM":            {"file": "lightgbm_multiclass.pkl",      "encoded": True},
}


def load_artifacts():
    """Load scaler and encoder from Stage 3."""
    scaler_path = os.path.join(STAGE3_MODELS, "final_scaler.pkl")
    encoder_path = os.path.join(STAGE3_MODELS, "stage3_multiclass_encoder.pkl")

    for p, name in [(scaler_path, "scaler"), (encoder_path, "encoder")]:
        if not os.path.exists(p):
            logging.error(f"{name} not found at {p}")
            sys.exit(1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return scaler, encoder


def load_external_data(scaler, encoder):
    """Load, filter, sample, and scale the Lycos multiclass dataset."""
    data_path = os.path.join(STAGE2_ARTIFACTS, "lycos_multiclass.parquet")
    print(f"\nLoading Lycos multiclass dataset from {data_path} ...")
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    print(f"  Total rows: {len(df):,}")
    print(f"  Labels present: {df['Label'].unique().tolist()}")

    # Filter to only labels the encoder knows
    known_classes = set(encoder.classes_)
    original_len = len(df)
    df = df[df["Label"].isin(known_classes)].copy()
    filtered_out = original_len - len(df)
    if filtered_out > 0:
        print(f"  Filtered out {filtered_out:,} rows with unknown labels")

    # Stratified sample for efficiency
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
        logging.error(f"Missing features in Lycos: {missing}")
        sys.exit(1)

    X_scaled = scaler.transform(X[scaler_cols])
    print(f"  Features scaled: {X_scaled.shape[1]}")
    print(f"  Final label distribution:")
    for lbl, cnt in zip(*np.unique(y_str, return_counts=True)):
        print(f"    {lbl}: {cnt:,}")

    return X_scaled, y_str


def evaluate_model(name, meta, X_scaled, y_str, encoder):
    """Evaluate a single model and return metrics dict."""
    pkl_path = os.path.join(STAGE3_MODELS, meta["file"])
    if not os.path.exists(pkl_path):
        logging.warning(f"Model not found: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    if meta["encoded"]:
        y_true = encoder.transform(y_str)
        X_eval = X_scaled
    else:
        y_true = y_str
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
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    auroc = 0.0
    if y_prob is not None:
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auroc = 0.0

    print(f"    Accuracy:  {acc:.6f}")
    print(f"    Macro-F1:  {f1:.6f}")
    print(f"    MCC:       {mcc:.6f}")
    print(f"    AUROC:     {auroc:.6f}")
    print(f"    Pred time: {pred_time:.4f}s")

    # Classification report — labels from actual data
    all_labels = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
    if meta["encoded"]:
        target_names = [encoder.inverse_transform([l])[0] for l in all_labels]
    else:
        target_names = [str(l) for l in all_labels]

    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, target_names=target_names,
        output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "Class"})
    report_path = os.path.join(TABLES_DIR, f"lycos_multiclass_{name.lower().replace(' ', '_')}_report.csv")
    df_report.to_csv(report_path, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Lycos Multiclass — {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"cm_lycos_multi_{name.lower().replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    logging.info(f"{name}: Acc={acc:.6f}, F1={f1:.6f}, MCC={mcc:.6f}, AUROC={auroc:.6f}")

    return {
        "Dataset": "Lycos",
        "Task_Type": "multiclass",
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "F1_Weighted": f1_weighted,
        "ROC_AUC": auroc,
        "MCC": mcc,
        "Prediction_Time_sec": pred_time,
        "Samples_Evaluated": len(X_eval)
    }


def main():
    print("=" * 60)
    print(" STAGE 4: LYCOS MULTICLASS GENERALIZATION EVALUATION ")
    print("=" * 60)

    scaler, encoder = load_artifacts()
    X_scaled, y_str = load_external_data(scaler, encoder)

    results = []
    for name, meta in MODELS.items():
        row = evaluate_model(name, meta, X_scaled, y_str, encoder)
        if row:
            results.append(row)

    df_results = pd.DataFrame(results)
    out_path = os.path.join(TABLES_DIR, "lycos_multiclass_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\n" + df_results.to_string(index=False))

    logging.info("Script 02 complete — Lycos multiclass evaluation finished.")
    print("\nDone!")


if __name__ == "__main__":
    main()
