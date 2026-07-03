"""
Stage 4 — Script 09: Unseen Attack Analysis
=============================================
Identifies attack types in external datasets (CIC2018, Lycos) that were
NEVER seen during CICIDS2017 training, and profiles how models handle
these truly novel attacks.

Since Stage 2 already mapped labels to families, "unseen" means a
family-level label in the external dataset that does NOT exist in the
CICIDS2017 training set (e.g., "OTHER" in CIC2018).
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
    filename=os.path.join(LOGS_DIR, "09_unseen_attack_analysis.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ── Model registry (multiclass) ─────────────────────────────────────
MODELS_MULTI = {
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


def analyze_unseen_attacks(dataset_name, parquet_file, scaler, encoder):
    """Identify and profile unseen attack labels."""
    data_path = os.path.join(STAGE2_ARTIFACTS, parquet_file)
    if not os.path.exists(data_path):
        logging.warning(f"Dataset not found: {data_path}")
        return [], []

    print(f"\n{'-'*50}")
    print(f"Analyzing unseen attacks in {dataset_name} ...")

    df = pd.read_parquet(data_path)

    # Training labels (what the encoder knows)
    known_labels = set(encoder.classes_)
    all_labels = set(df["Label"].unique())

    # Unseen = labels NOT in training AND not BENIGN
    unseen_labels = all_labels - known_labels
    # Remove BENIGN from unseen if present (it's always known)
    unseen_labels.discard("BENIGN")

    print(f"  Total unique labels: {len(all_labels)}")
    print(f"  Known (training) labels: {known_labels}")
    print(f"  Unseen attack labels: {unseen_labels}")

    if not unseen_labels:
        print("  No unseen attacks found in this dataset.")
        return [], []

    X = df.drop(columns=["Label"])
    y_str = np.array(df["Label"].astype(str))

    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else X.columns.tolist()
    X_scaled = scaler.transform(X[scaler_cols])

    results = []
    profiles = []

    for unseen_label in sorted(unseen_labels):
        mask = (y_str == unseen_label)
        n_samples = mask.sum()
        print(f"\n  Unseen label: '{unseen_label}' ({n_samples:,} samples)")

        X_unseen = X_scaled[mask]

        for model_name, meta in MODELS_MULTI.items():
            pkl_path = os.path.join(STAGE3_MODELS, meta["file"])
            if not os.path.exists(pkl_path):
                continue

            with open(pkl_path, "rb") as f:
                model = pickle.load(f)

            y_pred_raw = model.predict(X_unseen)

            if meta["encoded"]:
                y_pred_str = encoder.inverse_transform(y_pred_raw)
            else:
                y_pred_str = np.array(y_pred_raw)

            unique_preds, counts = np.unique(y_pred_str, return_counts=True)
            pred_dist = dict(zip(unique_preds, counts))

            benign_count = pred_dist.get("BENIGN", 0)
            attack_detected = n_samples - benign_count
            detection_rate = attack_detected / n_samples if n_samples > 0 else 0.0

            mean_confidence = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_unseen)
                    mean_confidence = float(np.mean(np.max(probs, axis=1)))
                except Exception:
                    pass

            most_common_pred = unique_preds[np.argmax(counts)]

            results.append({
                "Dataset": dataset_name,
                "Unseen_Label": unseen_label,
                "Samples": n_samples,
                "Model": model_name,
                "Attack_Detection_Rate": detection_rate,
                "Most_Common_Prediction": most_common_pred,
                "Predicted_BENIGN": benign_count,
                "Predicted_ATTACK": attack_detected,
                "Mean_Confidence": mean_confidence,
            })

            for pred_label, count in sorted(pred_dist.items(), key=lambda x: -x[1]):
                profiles.append({
                    "Dataset": dataset_name,
                    "Unseen_Label": unseen_label,
                    "Model": model_name,
                    "Predicted_As": pred_label,
                    "Count": int(count),
                    "Percentage": count / n_samples * 100 if n_samples > 0 else 0.0,
                })

            print(f"    {model_name}: Detection={detection_rate:.3f}, "
                  f"Most common -> '{most_common_pred}', "
                  f"Confidence={mean_confidence:.3f}")

    return results, profiles


def main():
    print("=" * 60)
    print(" STAGE 4: UNSEEN ATTACK ANALYSIS ")
    print("=" * 60)

    scaler, encoder = load_artifacts()

    all_results = []
    all_profiles = []

    for ds_name, ds_file in DATASETS.items():
        res, prof = analyze_unseen_attacks(ds_name, ds_file, scaler, encoder)
        all_results.extend(res)
        all_profiles.extend(prof)

    if all_results:
        df_results = pd.DataFrame(all_results)
        out_path = os.path.join(TABLES_DIR, "unseen_attack_results.csv")
        df_results.to_csv(out_path, index=False)
        print(f"\nUnseen attack results saved to {out_path}")

        print("\nUnseen Attack Detection Summary:")
        summary = df_results.groupby("Model").agg(
            Avg_Detection_Rate=("Attack_Detection_Rate", "mean"),
            Avg_Confidence=("Mean_Confidence", "mean"),
        ).reset_index().sort_values("Avg_Detection_Rate", ascending=False)
        print(summary.to_string(index=False))
    else:
        print("\nNo unseen attacks detected in any external dataset.")
        pd.DataFrame(columns=[
            "Dataset", "Unseen_Label", "Samples", "Model",
            "Attack_Detection_Rate", "Most_Common_Prediction",
            "Predicted_BENIGN", "Predicted_ATTACK", "Mean_Confidence"
        ]).to_csv(os.path.join(TABLES_DIR, "unseen_attack_results.csv"), index=False)

    if all_profiles:
        df_profiles = pd.DataFrame(all_profiles)
        prof_path = os.path.join(TABLES_DIR, "unseen_attack_profile.csv")
        df_profiles.to_csv(prof_path, index=False)
        print(f"Unseen attack profiles saved to {prof_path}")

        for dataset in df_profiles["Dataset"].unique():
            df_ds = df_profiles[df_profiles["Dataset"] == dataset]
            for model_name in df_ds["Model"].unique():
                df_m = df_ds[df_ds["Model"] == model_name]
                pivot = df_m.pivot_table(
                    index="Unseen_Label", columns="Predicted_As",
                    values="Percentage", aggfunc="mean", fill_value=0
                )
                if pivot.empty:
                    continue

                plt.figure(figsize=(max(10, len(pivot.columns)*1.2), max(4, len(pivot)*0.8)))
                sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrBr",
                            linewidths=0.5, vmin=0, vmax=100)
                plt.title(f"Unseen Attack Misclassification — {dataset} / {model_name}")
                plt.xlabel("Predicted As")
                plt.ylabel("True Unseen Label")
                plt.tight_layout()
                safe_name = model_name.lower().replace(" ", "_")
                plt.savefig(os.path.join(FIGURES_DIR,
                            f"unseen_profile_{dataset.lower()}_{safe_name}.png"), dpi=300)
                plt.close()
    else:
        pd.DataFrame(columns=[
            "Dataset", "Unseen_Label", "Model", "Predicted_As", "Count", "Percentage"
        ]).to_csv(os.path.join(TABLES_DIR, "unseen_attack_profile.csv"), index=False)

    logging.info("Script 09 complete — unseen attack analysis finished.")
    print("\nDone!")


if __name__ == "__main__":
    main()
