"""
Stage 4 — Script 07: Model Robustness Ranking
===============================================
Aggregates all Stage 4 results (multiclass + binary × CIC2018 + Lycos)
and ranks models by average F1, MCC, and AUROC across external datasets.
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

for d in [TABLES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "07_model_robustness.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_all_results():
    """Load and concatenate all external evaluation results."""
    result_files = [
        "cic2018_multiclass_results.csv",
        "lycos_multiclass_results.csv",
        "binary_transfer_results.csv",
    ]

    dfs = []
    for fname in result_files:
        path = os.path.join(TABLES_DIR, fname)
        if not os.path.exists(path):
            logging.warning(f"Results file not found: {path}")
            continue
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded {fname}: {len(df)} rows")

    if not dfs:
        logging.error("No external results found.")
        sys.exit(1)

    return pd.concat(dfs, ignore_index=True)


def compute_robustness_ranking(df_all):
    """Compute per-model average metrics and rank."""
    # Group by model
    agg = df_all.groupby("Model").agg(
        Avg_F1=("F1_Score", "mean"),
        Avg_MCC=("MCC", "mean"),
        Avg_AUROC=("ROC_AUC", "mean"),
        Avg_Accuracy=("Accuracy", "mean"),
        Avg_Precision=("Precision", "mean"),
        Avg_Recall=("Recall", "mean"),
        Num_Evaluations=("F1_Score", "count"),
        Min_F1=("F1_Score", "min"),
        Max_F1=("F1_Score", "max"),
        Std_F1=("F1_Score", "std"),
    ).reset_index()

    # Composite robustness score (weighted average)
    # F1 50%, MCC 30%, AUROC 20%
    agg["Robustness_Score"] = (
        0.50 * agg["Avg_F1"] +
        0.30 * agg["Avg_MCC"] +
        0.20 * agg["Avg_AUROC"]
    )

    # Stability score (lower std = more stable)
    agg["Stability"] = 1.0 - agg["Std_F1"].fillna(0)
    agg["Stability"] = agg["Stability"].clip(0, 1)

    # Rank by robustness score
    agg = agg.sort_values("Robustness_Score", ascending=False).reset_index(drop=True)
    agg["Rank"] = range(1, len(agg) + 1)

    return agg


def compute_per_task_ranking(df_all):
    """Compute per-model, per-task_type rankings."""
    agg = df_all.groupby(["Model", "Task_Type"]).agg(
        Avg_F1=("F1_Score", "mean"),
        Avg_MCC=("MCC", "mean"),
        Avg_AUROC=("ROC_AUC", "mean"),
    ).reset_index()

    agg["Robustness_Score"] = (
        0.50 * agg["Avg_F1"] +
        0.30 * agg["Avg_MCC"] +
        0.20 * agg["Avg_AUROC"]
    )

    agg = agg.sort_values(["Task_Type", "Robustness_Score"], ascending=[True, False])
    return agg


def main():
    print("=" * 60)
    print(" STAGE 4: MODEL ROBUSTNESS RANKING ")
    print("=" * 60)

    print("\nLoading all external evaluation results...")
    df_all = load_all_results()
    print(f"\nTotal evaluations: {len(df_all)}")

    # Overall ranking
    print("\n" + "-" * 50)
    print("OVERALL ROBUSTNESS RANKING:")
    df_ranking = compute_robustness_ranking(df_all)

    rank_path = os.path.join(TABLES_DIR, "robustness_ranking.csv")
    df_ranking.to_csv(rank_path, index=False)
    print(f"\nRobustness ranking saved to {rank_path}")

    print("\n" + df_ranking[["Rank", "Model", "Avg_F1", "Avg_MCC",
                            "Avg_AUROC", "Robustness_Score", "Stability"]].to_string(index=False))

    # Per-task ranking
    print("\n" + "-" * 50)
    print("PER-TASK ROBUSTNESS:")
    df_per_task = compute_per_task_ranking(df_all)
    per_task_path = os.path.join(TABLES_DIR, "robustness_per_task.csv")
    df_per_task.to_csv(per_task_path, index=False)
    print("\n" + df_per_task.to_string(index=False))

    # Also save the consolidated cross-dataset results
    cross_path = os.path.join(TABLES_DIR, "cross_dataset_results.csv")
    df_all.to_csv(cross_path, index=False)
    print(f"\nConsolidated cross-dataset results saved to {cross_path}")

    # Winner announcement
    if not df_ranking.empty:
        winner = df_ranking.iloc[0]
        print(f"\nWinner Most Robust Model: {winner['Model']} "
              f"(Score: {winner['Robustness_Score']:.4f}, "
              f"Avg F1: {winner['Avg_F1']:.4f})")

    logging.info("Script 07 complete — model robustness ranking finished.")
    print("\nDone!")


if __name__ == "__main__":
    main()
