"""
Stage 4 — Script 08: Comprehensive Visualization
==================================================
Generates all Stage 4 publication-quality figures:

1. F1 comparison (internal vs external per model)
2. AUROC comparison
3. Performance degradation bar chart
4. Model robustness ranking
5. Domain shift violin/box plots
6. Attack family transfer heatmap
7. Binary vs Multiclass transfer comparison
"""

import os
import sys
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

STAGE3_TABLES = os.path.abspath(os.path.join(STAGE4_DIR, "../stage3/tables"))

for d in [FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "08_visualization.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ── Plot styling ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})
PALETTE = sns.color_palette("Set2", 8)


def safe_load(filename, directory=TABLES_DIR):
    """Load a CSV if it exists, else return empty DataFrame."""
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    logging.warning(f"File not found: {path}")
    return pd.DataFrame()


def plot_f1_comparison():
    """Bar chart: Internal vs External F1 per model."""
    print("  Generating F1 comparison plot ...")
    df_int = safe_load("model_comparison.csv", STAGE3_TABLES)
    df_ext = safe_load("cross_dataset_results.csv")

    if df_int.empty or df_ext.empty:
        print("    Skipped (missing data)")
        return

    # Multiclass comparison
    df_int_multi = df_int[df_int["Task_Type"] == "multiclass"][["Model", "F1_Score"]].copy()
    df_int_multi = df_int_multi.rename(columns={"F1_Score": "Internal_F1"})

    df_ext_multi = df_ext[df_ext["Task_Type"] == "multiclass"].groupby("Model")["F1_Score"].mean().reset_index()
    df_ext_multi = df_ext_multi.rename(columns={"F1_Score": "External_F1"})

    df_cmp = df_int_multi.merge(df_ext_multi, on="Model", how="inner")
    if df_cmp.empty:
        return

    df_melt = df_cmp.melt(id_vars="Model", value_vars=["Internal_F1", "External_F1"],
                          var_name="Evaluation", value_name="F1_Score")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_melt, x="Model", y="F1_Score", hue="Evaluation",
                     palette=["#2ecc71", "#e74c3c"])
    plt.title("Stage 4: Internal vs External Multiclass F1-Score")
    plt.ylabel("Macro F1-Score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "f1_internal_vs_external.png"), dpi=300)
    plt.close()


def plot_auroc_comparison():
    """Bar chart: Internal vs External AUROC per model."""
    print("  Generating AUROC comparison plot ...")
    df_int = safe_load("model_comparison.csv", STAGE3_TABLES)
    df_ext = safe_load("cross_dataset_results.csv")

    if df_int.empty or df_ext.empty:
        print("    Skipped (missing data)")
        return

    df_int_multi = df_int[df_int["Task_Type"] == "multiclass"][["Model", "ROC_AUC"]].copy()
    df_int_multi = df_int_multi.rename(columns={"ROC_AUC": "Internal_AUROC"})

    df_ext_multi = df_ext[df_ext["Task_Type"] == "multiclass"].groupby("Model")["ROC_AUC"].mean().reset_index()
    df_ext_multi = df_ext_multi.rename(columns={"ROC_AUC": "External_AUROC"})

    df_cmp = df_int_multi.merge(df_ext_multi, on="Model", how="inner")
    if df_cmp.empty:
        return

    df_melt = df_cmp.melt(id_vars="Model", value_vars=["Internal_AUROC", "External_AUROC"],
                          var_name="Evaluation", value_name="AUROC")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x="Model", y="AUROC", hue="Evaluation",
                palette=["#3498db", "#e67e22"])
    plt.title("Stage 4: Internal vs External Multiclass AUROC")
    plt.ylabel("AUROC")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "auroc_internal_vs_external.png"), dpi=300)
    plt.close()


def plot_degradation():
    """Bar chart: F1 performance degradation per model per dataset."""
    print("  Generating degradation plot ...")
    df = safe_load("degradation_report.csv")
    if df.empty:
        print("    Skipped (missing data)")
        return

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x="Model", y="F1_Drop_Pct", hue="External_Dataset",
                palette="RdYlGn_r")
    plt.title("Stage 4: F1-Score Performance Degradation (%)")
    plt.ylabel("F1 Drop (%)")
    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="External Dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "f1_degradation.png"), dpi=300)
    plt.close()


def plot_robustness_ranking():
    """Horizontal bar chart: Model robustness scores."""
    print("  Generating robustness ranking plot ...")
    df = safe_load("robustness_ranking.csv")
    if df.empty:
        print("    Skipped (missing data)")
        return

    df = df.sort_values("Robustness_Score", ascending=True)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df))
    bars = plt.barh(df["Model"], df["Robustness_Score"], color=colors)
    plt.xlabel("Robustness Score")
    plt.title("Stage 4: Model Robustness Ranking (Cross-Dataset)")

    for bar, score in zip(bars, df["Robustness_Score"]):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{score:.4f}", va="center", fontsize=10)

    plt.xlim(0, df["Robustness_Score"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "robustness_ranking.png"), dpi=300)
    plt.close()


def plot_domain_shift():
    """Box plots showing feature drift severity across datasets."""
    print("  Generating domain shift plots ...")
    df = safe_load("feature_shift.csv")
    if df.empty:
        print("    Skipped (missing data)")
        return

    # PSI distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df, x="Dataset", y="PSI", ax=axes[0], palette="Set2")
    axes[0].set_title("Population Stability Index (PSI) by Dataset")
    axes[0].axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="Low/Moderate (0.1)")
    axes[0].axhline(y=0.25, color="red", linestyle="--", alpha=0.7, label="Moderate/High (0.25)")
    axes[0].legend(fontsize=9)

    # KS statistic distribution
    sns.boxplot(data=df, x="Dataset", y="KS_Statistic", ax=axes[1], palette="Set3")
    axes[1].set_title("KS Statistic by Dataset")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "domain_shift_distributions.png"), dpi=300)
    plt.close()

    # Drift severity counts
    drift_counts = df.groupby(["Dataset", "Drift_Severity"]).size().reset_index(name="Count")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=drift_counts, x="Dataset", y="Count", hue="Drift_Severity",
                palette={"LOW": "#2ecc71", "MODERATE": "#f39c12", "HIGH": "#e74c3c"})
    plt.title("Stage 4: Feature Drift Severity Distribution")
    plt.ylabel("Number of Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "drift_severity_counts.png"), dpi=300)
    plt.close()


def plot_family_heatmap():
    """Heatmap: Attack family transfer accuracy per model."""
    print("  Generating attack family heatmap ...")
    df = safe_load("family_results.csv")
    if df.empty:
        print("    Skipped (missing data)")
        return

    for dataset in df["Dataset"].unique():
        df_ds = df[df["Dataset"] == dataset]
        pivot = df_ds.pivot_table(index="Family", columns="Model",
                                  values="Attack_Detection_Rate", aggfunc="mean")

        if pivot.empty:
            continue

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=0, vmax=1, linewidths=0.5)
        plt.title(f"Stage 4: Attack Family Detection Rate — {dataset}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"family_heatmap_{dataset.lower()}.png"), dpi=300)
        plt.close()


def plot_binary_vs_multiclass():
    """Compare binary vs multiclass F1 on external datasets."""
    print("  Generating binary vs multiclass comparison ...")
    df = safe_load("cross_dataset_results.csv")
    if df.empty:
        print("    Skipped (missing data)")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="F1_Score", hue="Task_Type",
                palette=["#9b59b6", "#1abc9c"])
    plt.title("Stage 4: Binary vs Multiclass F1 on External Datasets")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Task")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "binary_vs_multiclass_external.png"), dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print(" STAGE 4: COMPREHENSIVE VISUALIZATION ")
    print("=" * 60)

    plot_f1_comparison()
    plot_auroc_comparison()
    plot_degradation()
    plot_robustness_ranking()
    plot_domain_shift()
    plot_family_heatmap()
    plot_binary_vs_multiclass()

    print(f"\nAll figures saved to {FIGURES_DIR}")
    logging.info("Script 08 complete — all visualizations generated.")
    print("\nDone!")


if __name__ == "__main__":
    main()
