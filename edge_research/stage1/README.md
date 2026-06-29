# STAGE 1 — Data Audit & Domain Shift Analysis

## 📌 Overview
**Stage 1** establishes a comprehensive, purely exploratory, and scientifically rigorous understanding of cross-dataset structural mismatch, domain drift, and topology artifact leakage across three foundational network intrusion datasets:
1. **CICIDS2017** (Baseline Training Distribution)
2. **CSE-CIC-IDS2018** (External Evaluation Set)
3. **Lycos-Unicas-IDS2018** (External Evaluation Set)

This stage mathematically proves that cross-dataset evaluation cannot be treated as a standard i.i.d. classification task due to severe covariate shift, topology leakage, and unseen threat categories.

---

## 📂 Directory Structure

```
stage1/
├── figures/                    # High-resolution empirical visualizations (PCA, t-SNE, KDEs)
├── logs/                       # Execution logs from analytical scripts
├── reports/                    # Comprehensive research reports (e.g., stage1_report.md)
├── scripts/                    # Sequentially numbered analysis scripts
├── tables/                     # Generated CSV inventories and shift score tables
└── README.md                   # Stage 1 Documentation (This File)
```

---

## 📜 Executable Script Pipeline

The `scripts/` directory contains 13 dedicated, sequentially numbered Python modules designed to execute the complete Stage 1 auditing workflow:

| Script Name | Description & Objective |
| :--- | :--- |
| `01_dataset_inventory.py` | Audits row/column counts, memory footprints, and disk size across datasets. |
| `02_class_distribution.py` | Analyzes class frequencies, percentage compositions, and log-scaled long-tail distributions. |
| `03_class_overlap.py` | Maps shared benign/attack classes versus unseen, zero-day threat families in external sets. |
| `04_feature_inventory.py` | Inspects feature overlap, identifying exact matches, renamed equivalents, and missing columns. |
| `05_missing_values.py` | Quantifies missing values, NaNs, and unrecorded flow properties. |
| `06_feature_statistics.py` | Calculates baseline statistical moments (mean, std, min, max, quartiles) for all features. |
| `07_domain_shift_analysis.py` | Performs empirical domain shift evaluations via comparative histograms and KDEs. |
| `08_correlation_analysis.py` | Generates full 70x70 feature correlation matrices to evaluate multi-collinearity changes. |
| `09_topology_features.py` | Investigates confounding topology artifacts (`Source Port`, `Timestamp`, `Destination Port`). |
| `10_dataset_shift_metrics.py` | Quantifies feature drift using Mean Jensen-Shannon divergence and Wasserstein distance. |
| `11_pca_visualization.py` | Projects high-dimensional feature spaces into 2D via Principal Component Analysis (PCA). |
| `12_tsne_visualization.py` | Captures non-linear manifold structures across corpora using t-SNE. |
| `13_summary_report.py` | Aggregates empirical tables and generates comprehensive summary reports. |

---

## 📊 Key Findings & Empirical Deliverables

1. **Lycos-Unicas-IDS2018 Structural Divergence:** Exhibits the most extreme domain shift, completely lacking foundational baseline features such as `Timestamp` and `Source Port`.
2. **Open-World Concept Shift:** External datasets introduce unrepresented zero-day attack variants (e.g., HOIC, LOIC-UDP, Web Brute Force, SQL Injection), causing severe generalization penalties in standard decision tree models.
3. **Severe Covariate Shift:** Fundamental flow properties like `Flow Duration`, `Flow Bytes/s`, and `Packet Length Mean/Std` exhibit major drift due to differing hardware testbed speeds and background traffic generators.
4. **Topology Artifact Leakage:** Models trained on CICIDS2017 often establish splitting criteria on ephemeral artifacts (`Source Port`, `Timestamp`) rather than generalizable invariant attack semantics.

For the full academic write-up, refer to [`reports/stage1_report.md`](reports/stage1_report.md).
