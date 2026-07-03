"""
Stage 4 — Script 04: Attack Family Mapping & Per-Family Transfer Evaluation
=============================================================================
The Stage 2 pipeline already mapped raw labels to canonical attack families.
This script evaluates per-family transfer performance across CIC2018 and Lycos.

The deterministic FAMILY_MAP is used to identify which raw labels from
CICIDS2017 correspond to each family (for provenance/documentation), but
since Stage 2 already produced family-level labels, we evaluate directly.
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Directory structure ──────────────────────────────────────────────
STAGE4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE4_DIR, "tables")
LOGS_DIR = os.path.join(STAGE4_DIR, "logs")

STAGE3_MODELS = os.path.abspath(os.path.join(STAGE4_DIR, "../stage3/models"))
STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE4_DIR, "../stage2/artifacts"))

for d in [TABLES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "04_attack_family_mapping.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ── Deterministic Family Map (raw → canonical) ──────────────────────
# Documents the original raw-to-family mapping used in Stage 2.
# Stage 2 parquet labels are ALREADY at the family level.
FAMILY_MAP = {
    "DoS Hulk": "DOS_DDOS",
    "DoS GoldenEye": "DOS_DDOS",
    "DoS Slowloris": "DOS_DDOS",
    "DoS slowloris": "DOS_DDOS",
    "DoS Slowhttptest": "DOS_DDOS",
    "DDoS": "DOS_DDOS",
    "PortScan": "PROBING",
    "FTP-Patator": "BRUTE_FORCE",
    "SSH-Patator": "BRUTE_FORCE",
    "Web Attack \u2013 Brute Force": "WEB_ATTACK",
    "Web Attack \u2013 XSS": "WEB_ATTACK",
    "Web Attack \u2013 Sql Injection": "WEB_ATTACK",
    "Bot": "BOT",
}

# Attack families (the labels as they appear in parquet files)
ATTACK_FAMILIES = ["DOS_DDOS", "PROBING", "BRUTE_FORCE", "WEB_ATTACK", "BOT"]

SAMPLE_SIZE = 500_000

# ── Model registry (multiclass) ─────────────────────────────────────
MODELS = {
    "Logistic Regression": {"file": "logistic_multiclass.pkl",      "encoded": False},
    "Random Forest":       {"file": "random_forest_multiclass.pkl", "encoded": False},
    "Extra Trees":         {"file": "extra_trees_multiclass.pkl",   "encoded": False},
    "XGBoost":             {"file": "xgboost_multiclass.pkl",       "encoded": True},
    "LightGBM":            {"file": "lightgbm_multiclass.pkl",      "encoded": True},
}

DATASETS = {
    "CIC2018": "cic2018_multiclass.parquet",
    "Lycos":   "lycos_multiclass.parquet",
}


def load_artifacts():
    """Load scaler and encoder from Stage 3."""
    scaler_path = os.path.join(STAGE3_MODELS, "final_scaler.pkl")
    encoder_path = os.path.join(STAGE3_MODELS, "stage3_multiclass_encoder.pkl")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return scaler, encoder


def evaluate_families(dataset_name, parquet_file, scaler, encoder):
    """Evaluate per-family transfer performance on one external dataset."""
    data_path = os.path.join(STAGE2_ARTIFACTS, parquet_file)
    print(f"\n{'-'*50}")
    print(f"Processing {dataset_name}: {data_path}")

    if not os.path.exists(data_path):
        logging.error(f"Dataset not found: {data_path}")
        return [], []

    df = pd.read_parquet(data_path)
    print(f"  Total rows: {len(df):,}")
    print(f"  Labels: {df['Label'].unique().tolist()}")

    # Stratified sample
    if len(df) > SAMPLE_SIZE:
        original_len = len(df)
        df_sampled = []
        for lbl, group in df.groupby("Label"):
            n_samples = min(len(group), max(100, int(SAMPLE_SIZE * len(group) / original_len)))
            df_sampled.append(group.sample(n_samples, random_state=42))
        df = pd.concat(df_sampled, ignore_index=True)
        print(f"  Stratified sample: {len(df):,} rows")

    X = df.drop(columns=["Label"])
    y_str = np.array(df["Label"].astype(str))

    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else X.columns.tolist()
    X_scaled = scaler.transform(X[scaler_cols])

    # Build mapping table for output
    mapping_records = []
    for lbl in sorted(set(y_str)):
        mapping_records.append({
            "Dataset": dataset_name,
            "Label_In_Parquet": lbl,
            "Is_Attack_Family": lbl in ATTACK_FAMILIES,
            "Present_In_Training": lbl in set(encoder.classes_),
        })

    # Identify which families are in this external dataset
    ext_families = [f for f in ATTACK_FAMILIES if f in set(y_str)]
    print(f"  Attack families found: {ext_families}")

    results = []

    for model_name, meta in MODELS.items():
        pkl_path = os.path.join(STAGE3_MODELS, meta["file"])
        if not os.path.exists(pkl_path):
            continue

        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        # Get predictions
        y_pred_raw = model.predict(X_scaled)

        # Decode to strings
        if meta["encoded"]:
            y_pred_str = encoder.inverse_transform(y_pred_raw)
        else:
            y_pred_str = np.array(y_pred_raw)

        # Per-family evaluation
        for family in ext_families:
            family_mask = (y_str == family)
            n_samples = family_mask.sum()
            if n_samples == 0:
                continue

            y_true_fam = y_str[family_mask]
            y_pred_fam = y_pred_str[family_mask]

            # Exact family match accuracy
            correct = (y_true_fam == y_pred_fam).sum()
            family_acc = correct / n_samples

            # Attack detection rate: predicted as ANY attack (not BENIGN)
            not_benign = (y_pred_fam != "BENIGN").sum()
            attack_detection_rate = not_benign / n_samples

            # What was it most commonly predicted as?
            unique_preds, pred_counts = np.unique(y_pred_fam, return_counts=True)
            most_common = unique_preds[np.argmax(pred_counts)]

            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Family": family,
                "Samples": n_samples,
                "Family_Accuracy": family_acc,
                "Attack_Detection_Rate": attack_detection_rate,
                "Correct_Family": int(correct),
                "Detected_As_Attack": int(not_benign),
                "Most_Common_Prediction": most_common,
            })

        print(f"  {model_name}: family evaluation complete")

    return results, mapping_records


def main():
    print("=" * 60)
    print(" STAGE 4: ATTACK FAMILY MAPPING & TRANSFER EVALUATION ")
    print("=" * 60)

    scaler, encoder = load_artifacts()

    all_results = []
    all_mappings = []

    for ds_name, ds_file in DATASETS.items():
        res, maps = evaluate_families(ds_name, ds_file, scaler, encoder)
        all_results.extend(res)
        all_mappings.extend(maps)

    # Save family results
    df_results = pd.DataFrame(all_results)
    out_path = os.path.join(TABLES_DIR, "family_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nFamily results saved to {out_path}")

    # Save mapping table
    df_mapping = pd.DataFrame(all_mappings)
    map_path = os.path.join(TABLES_DIR, "family_mapping.csv")
    df_mapping.to_csv(map_path, index=False)
    print(f"Family mapping saved to {map_path}")

    # Also save the raw→family reference
    ref_records = [{"Raw_Label": k, "Family": v} for k, v in FAMILY_MAP.items()]
    pd.DataFrame(ref_records).to_csv(os.path.join(TABLES_DIR, "family_map_reference.csv"), index=False)

    if not df_results.empty:
        print("\nFamily Transfer Results Summary:")
        print(df_results.to_string(index=False))

    logging.info("Script 04 complete — attack family mapping & evaluation finished.")
    print("\nDone!")


if __name__ == "__main__":
    main()
