"""
Stage 4 — Script 05: Performance Degradation Analysis
======================================================
Computes the percentage performance drop from internal (Stage 3) to
external (Stage 4) evaluation for every model.

Formula: ((Internal_F1 - External_F1) / Internal_F1) × 100
"""

import os
import sys
import logging
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ── Directory structure ──────────────────────────────────────────────
STAGE4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(STAGE4_DIR, "tables")
LOGS_DIR = os.path.join(STAGE4_DIR, "logs")

STAGE3_TABLES = os.path.abspath(os.path.join(STAGE4_DIR, "../stage3/tables"))

for d in [TABLES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "05_performance_degradation.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_internal_results():
    """Load Stage 3 model comparison results (internal baseline)."""
    path = os.path.join(STAGE3_TABLES, "model_comparison.csv")
    if not os.path.exists(path):
        logging.error(f"Stage 3 model_comparison.csv not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded internal results: {len(df)} rows from {path}")
    return df


def load_external_results():
    """Load all Stage 4 external evaluation results."""
    files = {
        "cic2018_multiclass_results.csv": ("CIC2018", "multiclass"),
        "lycos_multiclass_results.csv":   ("Lycos",   "multiclass"),
        "binary_transfer_results.csv":    (None,      None),  # contains both datasets
    }

    dfs = []
    for fname, (ds, tt) in files.items():
        path = os.path.join(TABLES_DIR, fname)
        if not os.path.exists(path):
            logging.warning(f"External results not found: {path}")
            continue
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"Loaded external results: {len(df)} rows from {fname}")

    if not dfs:
        logging.error("No external results found.")
        sys.exit(1)

    return pd.concat(dfs, ignore_index=True)


def compute_degradation(df_internal, df_external):
    """Compute F1 degradation for each model × task_type combination."""
    records = []

    for _, ext_row in df_external.iterrows():
        model = ext_row["Model"]
        task = ext_row["Task_Type"]
        dataset = ext_row.get("Dataset", "Unknown")
        ext_f1 = ext_row["F1_Score"]
        ext_acc = ext_row.get("Accuracy", None)
        ext_mcc = ext_row.get("MCC", None)
        ext_auroc = ext_row.get("ROC_AUC", None)

        # Find matching internal result
        int_match = df_internal[
            (df_internal["Model"] == model) &
            (df_internal["Task_Type"] == task)
        ]

        if int_match.empty:
            logging.warning(f"No internal match for {model}/{task}")
            continue

        int_row = int_match.iloc[0]
        int_f1 = int_row["F1_Score"]
        int_acc = int_row.get("Accuracy", None)
        int_mcc = int_row.get("MCC", None)
        int_auroc = int_row.get("ROC_AUC", None)

        # F1 degradation
        f1_drop_pct = ((int_f1 - ext_f1) / int_f1 * 100) if int_f1 > 0 else 0.0

        # Accuracy degradation
        acc_drop_pct = 0.0
        if int_acc and int_acc > 0 and ext_acc is not None:
            acc_drop_pct = ((int_acc - ext_acc) / int_acc * 100)

        # MCC degradation
        mcc_drop = 0.0
        if int_mcc is not None and ext_mcc is not None:
            mcc_drop = int_mcc - ext_mcc

        # AUROC degradation
        auroc_drop = 0.0
        if int_auroc is not None and ext_auroc is not None:
            auroc_drop = int_auroc - ext_auroc

        records.append({
            "Model": model,
            "Task_Type": task,
            "External_Dataset": dataset,
            "Internal_F1": int_f1,
            "External_F1": ext_f1,
            "F1_Drop_Pct": f1_drop_pct,
            "Internal_Accuracy": int_acc,
            "External_Accuracy": ext_acc,
            "Accuracy_Drop_Pct": acc_drop_pct,
            "Internal_MCC": int_mcc,
            "External_MCC": ext_mcc,
            "MCC_Drop": mcc_drop,
            "Internal_AUROC": int_auroc,
            "External_AUROC": ext_auroc,
            "AUROC_Drop": auroc_drop,
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print(" STAGE 4: PERFORMANCE DEGRADATION ANALYSIS ")
    print("=" * 60)

    df_internal = load_internal_results()
    df_external = load_external_results()

    df_degrad = compute_degradation(df_internal, df_external)

    out_path = os.path.join(TABLES_DIR, "degradation_report.csv")
    df_degrad.to_csv(out_path, index=False)
    print(f"\nDegradation report saved to {out_path}")

    print("\nPerformance Degradation Summary:")
    print(df_degrad[["Model", "Task_Type", "External_Dataset",
                     "Internal_F1", "External_F1", "F1_Drop_Pct"]].to_string(index=False))

    # Worst degradation
    if not df_degrad.empty:
        worst = df_degrad.loc[df_degrad["F1_Drop_Pct"].idxmax()]
        print(f"\nWorst F1 degradation: {worst['Model']} on {worst['External_Dataset']} "
              f"({worst['Task_Type']}): {worst['F1_Drop_Pct']:.2f}%")

        best = df_degrad.loc[df_degrad["F1_Drop_Pct"].idxmin()]
        print(f"Best F1 retention:   {best['Model']} on {best['External_Dataset']} "
              f"({best['Task_Type']}): {best['F1_Drop_Pct']:.2f}%")

    logging.info("Script 05 complete — performance degradation analysis finished.")
    print("\nDone!")


if __name__ == "__main__":
    main()
