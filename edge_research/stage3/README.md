# STAGE 3 — Baseline Modeling & Internal Validation

## 📌 Overview
**Stage 3** establishes the baseline machine learning performance and internal validation benchmarks for the network intrusion detection pipeline. Operating on the fully preprocessed and aligned data from Stage 2, this stage implements a suite of traditional and ensemble machine learning models to establish a highly optimized performance ceiling on the baseline distribution (CICIDS2017) before moving to cross-dataset generalization.

---

## 📂 Directory Structure

```
stage3/
├── figures/                    # ROC curves, Precision-Recall curves, and model comparison charts
├── logs/                       # Execution logs from training and evaluation scripts
├── models/                     # Saved binary model artifacts (.pkl or .onnx)
├── reports/                    # Comprehensive internal validation reports
├── scripts/                    # Sequentially numbered modeling scripts
├── tables/                     # Evaluation metrics, confusion matrices, and GPU benchmarks
└── README.md                   # Stage 3 Documentation (This File)
```

---

## 📜 Executable Script Pipeline

The `scripts/` directory contains 10 dedicated Python modules that execute the full baseline modeling, evaluation, and GPU benchmarking workflow:

| Script Name | Description & Objective |
| :--- | :--- |
| `01_dataset_loading.py` | Efficiently loads the highly optimized, memory-mapped datasets outputted by Stage 2. |
| `02_scaler_comparison.py` | Evaluates model sensitivity across different feature scaling techniques applied in Stage 2. |
| `03_logistic_regression.py` | Implements a linear baseline model (Logistic Regression) to gauge basic separability. |
| `04_random_forest.py` | Trains and tunes a standard Random Forest ensemble for robust baseline performance. |
| `05_xgboost.py` | Implements XGBoost, utilizing advanced gradient boosting for handling class imbalances. |
| `06_lightgbm.py` | Leverages LightGBM for highly efficient, histogram-based gradient boosting on large datasets. |
| `07_extra_trees.py` | Evaluates the Extremely Randomized Trees algorithm to reduce variance in tree splits. |
| `08_model_comparison.py` | Aggregates all model performances, plotting comparative ROC and PR curves. |
| `09_gpu_benchmark.py` | Benchmarks GPU-accelerated training times against CPU executions (e.g., cuML vs Scikit-Learn). |
| `10_internal_validation.py` | Executes rigorous k-fold cross-validation and internal holdout testing to prevent overfitting. |

---

## 🚀 Key Technical Highlights

- **Comprehensive Ensemble Evaluation:** Establishes rigorous baseline metrics across Logistic Regression, Random Forest, XGBoost, LightGBM, and Extra Trees, ensuring no single algorithmic bias is overlooked.
- **Hardware Acceleration:** Heavily leverages GPU benchmarks (`09_gpu_benchmark.py`) to quantify the training speedups achieved through hardware acceleration libraries (RAPIDS cuML, XGBoost-GPU) over standard CPU executions.
- **Internal Overfitting Prevention:** Implements stringent internal validation (`10_internal_validation.py`) to confirm that the models are learning generalizable representations of the baseline distribution, rather than simply memorizing the training set.

For detailed metric breakdowns, confusion matrices, and GPU speedup statistics, refer to [`reports/internal_validation_report.md`](reports/internal_validation_report.md).
