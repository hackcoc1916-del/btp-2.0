# STAGE 4 — Cross-Dataset Generalization Analysis

## 📌 Overview
**Stage 4** evaluates how well the Stage 3 baseline models (trained on CICIDS2017) generalize to completely unseen external datasets (CSE-CIC-IDS2018, Lycos-Unicas-IDS2018). This stage quantifies domain shift, measures model robustness, and identifies which models and attack families transfer successfully across network environments.

---

## 📂 Directory Structure

```
stage4/
├── figures/                    # Publication-quality visualizations and heatmaps
├── logs/                       # Execution logs from all analysis scripts
├── reports/                    # Comprehensive generalization report
├── scripts/                    # Sequentially numbered analysis scripts
├── tables/                     # Evaluation metrics, rankings, and drift analysis
└── README.md                   # Stage 4 Documentation (This File)
```

---

## 📜 Executable Script Pipeline

| Script Name | Description & Objective |
| :--- | :--- |
| `01_cic2018_multiclass.py` | Evaluates all 5 Stage 3 multiclass models on the CSE-CIC-IDS2018 external dataset. |
| `02_lycos_multiclass.py` | Evaluates all 5 Stage 3 multiclass models on the Lycos-Unicas-IDS2018 external dataset. |
| `03_binary_transfer.py` | Evaluates BENIGN vs ATTACK binary classification transfer on both external datasets. |
| `04_attack_family_mapping.py` | Maps multiclass labels into canonical attack families using a deterministic `FAMILY_MAP` and evaluates per-family transfer. |
| `05_performance_degradation.py` | Computes percentage performance drop from Stage 3 internal to Stage 4 external evaluation. |
| `06_domain_shift_analysis.py` | Analyzes feature drift (KS test), label drift (JSD), and distribution shift (PSI) between training and external datasets. |
| `07_model_robustness.py` | Ranks models by average F1, MCC, AUROC across all external evaluations; generates composite robustness score. |
| `08_visualization.py` | Generates all publication-quality figures: F1 comparison, AUROC, degradation, robustness ranking, domain shift, family heatmaps. |
| `09_unseen_attack_analysis.py` | Identifies and profiles attack types in external datasets never seen during CICIDS2017 training. |

---

## 🔬 Research Questions

| ID | Research Question |
| :--- | :--- |
| **RQ4.1** | How severe is domain shift between CICIDS2017 and external datasets? |
| **RQ4.2** | Which model generalizes best to unseen network environments? |
| **RQ4.3** | Does binary classification improve cross-dataset transfer? |
| **RQ4.4** | Which attack families transfer successfully across datasets? |
| **RQ4.5** | How much performance is lost when moving from internal to external evaluation? |

---

## 🚀 Key Technical Highlights

- **Deterministic Attack Family Mapping:** Uses a hardcoded `FAMILY_MAP` dictionary for fully reproducible, reviewer-friendly label-to-family assignments.
- **Statistical Domain Shift Quantification:** Employs Kolmogorov-Smirnov tests, Population Stability Index (PSI), and Jensen-Shannon Divergence for rigorous drift measurement.
- **Composite Robustness Scoring:** Models are ranked using a weighted combination of F1 (50%), MCC (30%), and AUROC (20%) across all external evaluations.
- **Unseen Attack Profiling:** Identifies novel attack types and measures how frequently models correctly flag them as non-benign traffic.
- **GPU Acceleration:** XGBoost and LightGBM inference leverage CUDA/GPU with automatic CPU fallback.

---

## 📊 Output Artifacts

| File | Description |
| :--- | :--- |
| `tables/cic2018_multiclass_results.csv` | CIC2018 multiclass evaluation metrics per model |
| `tables/lycos_multiclass_results.csv` | Lycos multiclass evaluation metrics per model |
| `tables/binary_transfer_results.csv` | Binary transfer results for both external datasets |
| `tables/family_results.csv` | Per-family transfer performance |
| `tables/family_mapping.csv` | Label-to-family mapping reference |
| `tables/degradation_report.csv` | Performance degradation percentages |
| `tables/feature_shift.csv` | Feature drift statistics (KS, PSI) |
| `tables/label_drift.csv` | Label distribution shift (JSD) |
| `tables/robustness_ranking.csv` | Model robustness ranking |
| `tables/cross_dataset_results.csv` | Consolidated cross-dataset results |
| `tables/unseen_attack_results.csv` | Unseen attack detection analysis |
| `tables/unseen_attack_profile.csv` | Misclassification profiles for unseen attacks |
| `reports/generalization_report.md` | Comprehensive Stage 4 report |

For detailed results, see [`reports/generalization_report.md`](reports/generalization_report.md).
