"""
Stage 4 — Script 06: Domain Shift Analysis
============================================
Statistically analyzes feature drift, label drift, and distribution shift
between the CICIDS2017 training set and external datasets (CIC2018, Lycos).

Methods:
  - Feature drift:       Kolmogorov-Smirnov test per feature
  - Label drift:         Jensen-Shannon divergence on class proportions
  - Distribution shift:  Population Stability Index (PSI) per feature
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Directory structure ──────────────────────────────────────────────
STAGE4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE4_DIR, "tables")
LOGS_DIR = os.path.join(STAGE4_DIR, "logs")

STAGE2_ARTIFACTS = os.path.abspath(os.path.join(STAGE4_DIR, "../stage2/artifacts"))

for d in [TABLES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "06_domain_shift_analysis.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

SAMPLE_SIZE = 50000  # rows to sample for efficiency

EXTERNAL_DATASETS = {
    "CIC2018": "cic2018_multiclass.parquet",
    "Lycos":   "lycos_multiclass.parquet",
}


def compute_psi(reference, test, bins=10):
    """Compute Population Stability Index between two distributions."""
    eps = 1e-10
    # Use reference quantiles for binning
    try:
        breakpoints = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    except Exception:
        return 0.0

    if len(breakpoints) <= 1:
        return 0.0

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    test_counts = np.histogram(test, bins=breakpoints)[0]

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    test_pct = test_counts / (test_counts.sum() + eps)

    ref_pct = np.clip(ref_pct, eps, 1)
    test_pct = np.clip(test_pct, eps, 1)

    psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
    return psi


def analyze_feature_drift(df_train, df_ext, dataset_name, feature_cols):
    """Run KS test and PSI for each feature."""
    records = []
    n_train = len(df_train)
    n_ext = len(df_ext)

    for col in feature_cols:
        train_vals = df_train[col].dropna().values
        ext_vals = df_ext[col].dropna().values

        if len(train_vals) == 0 or len(ext_vals) == 0:
            continue

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(train_vals, ext_vals)

        # PSI
        psi = compute_psi(train_vals, ext_vals, bins=10)

        # Basic stats
        mean_diff = abs(np.mean(ext_vals) - np.mean(train_vals))
        std_diff = abs(np.std(ext_vals) - np.std(train_vals))

        # Drift severity label
        if psi < 0.1:
            severity = "LOW"
        elif psi < 0.25:
            severity = "MODERATE"
        else:
            severity = "HIGH"

        records.append({
            "Dataset": dataset_name,
            "Feature": col,
            "KS_Statistic": ks_stat,
            "KS_PValue": ks_pval,
            "PSI": psi,
            "Drift_Severity": severity,
            "Mean_Diff": mean_diff,
            "Std_Diff": std_diff,
            "Train_Mean": np.mean(train_vals),
            "Ext_Mean": np.mean(ext_vals),
            "Train_Std": np.std(train_vals),
            "Ext_Std": np.std(ext_vals),
        })

    return pd.DataFrame(records)


def analyze_label_drift(df_train, df_ext, dataset_name):
    """Compute Jensen-Shannon divergence on label distributions."""
    train_dist = df_train["Label"].value_counts(normalize=True)
    ext_dist = df_ext["Label"].value_counts(normalize=True)

    # Align labels
    all_labels = sorted(set(train_dist.index) | set(ext_dist.index))
    p = np.array([train_dist.get(l, 0.0) for l in all_labels])
    q = np.array([ext_dist.get(l, 0.0) for l in all_labels])

    # Jensen-Shannon divergence
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    records = []
    for label in all_labels:
        records.append({
            "Dataset": dataset_name,
            "Label": label,
            "Train_Proportion": train_dist.get(label, 0.0),
            "External_Proportion": ext_dist.get(label, 0.0),
            "Proportion_Diff": abs(train_dist.get(label, 0.0) - ext_dist.get(label, 0.0)),
        })

    # Add summary row
    records.append({
        "Dataset": dataset_name,
        "Label": "__JSD_SUMMARY__",
        "Train_Proportion": jsd,
        "External_Proportion": jsd,
        "Proportion_Diff": jsd,
    })

    return pd.DataFrame(records), jsd


def main():
    print("=" * 60)
    print(" STAGE 4: DOMAIN SHIFT ANALYSIS ")
    print("=" * 60)

    # Load training data (CICIDS2017)
    train_path = os.path.join(STAGE2_ARTIFACTS, "cic2017_multiclass.parquet")
    if not os.path.exists(train_path):
        logging.error(f"Training dataset not found: {train_path}")
        sys.exit(1)

    print(f"\nLoading training data: {train_path}")
    df_train_full = pd.read_parquet(train_path)
    df_train = df_train_full.sample(min(len(df_train_full), SAMPLE_SIZE), random_state=42)
    feature_cols = [c for c in df_train.columns if c != "Label"]
    print(f"  Sampled {len(df_train):,} rows, {len(feature_cols)} features")

    all_feature_shift = []
    all_label_drift = []

    for ds_name, ds_file in EXTERNAL_DATASETS.items():
        ext_path = os.path.join(STAGE2_ARTIFACTS, ds_file)
        if not os.path.exists(ext_path):
            logging.warning(f"External dataset not found: {ext_path}")
            continue

        print(f"\n{'-'*50}")
        print(f"Analyzing domain shift: CICIDS2017 -> {ds_name}")
        df_ext_full = pd.read_parquet(ext_path)
        df_ext = df_ext_full.sample(min(len(df_ext_full), SAMPLE_SIZE), random_state=42)
        print(f"  Sampled {len(df_ext):,} rows")

        # Feature drift
        print("  Computing feature drift (KS + PSI) ...")
        df_feat = analyze_feature_drift(df_train, df_ext, ds_name, feature_cols)
        all_feature_shift.append(df_feat)

        high_drift = df_feat[df_feat["Drift_Severity"] == "HIGH"]
        print(f"  Features with HIGH drift: {len(high_drift)}/{len(feature_cols)}")

        # Label drift
        print("  Computing label drift (JSD) ...")
        df_lbl, jsd = analyze_label_drift(df_train, df_ext, ds_name)
        all_label_drift.append(df_lbl)
        print(f"  Jensen-Shannon Divergence: {jsd:.6f}")

        logging.info(f"{ds_name}: HIGH drift features={len(high_drift)}, JSD={jsd:.6f}")

    # Save results
    if all_feature_shift:
        df_fs = pd.concat(all_feature_shift, ignore_index=True)
        fs_path = os.path.join(TABLES_DIR, "feature_shift.csv")
        df_fs.to_csv(fs_path, index=False)
        print(f"\nFeature shift analysis saved to {fs_path}")

    if all_label_drift:
        df_ld = pd.concat(all_label_drift, ignore_index=True)
        ld_path = os.path.join(TABLES_DIR, "label_drift.csv")
        df_ld.to_csv(ld_path, index=False)
        print(f"Label drift analysis saved to {ld_path}")

    logging.info("Script 06 complete — domain shift analysis finished.")
    print("\nDone!")


if __name__ == "__main__":
    main()
