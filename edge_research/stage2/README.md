# STAGE 2 — Preprocessing, Feature Alignment & Pipeline Validation

## 📌 Overview
**Stage 2** implements a robust, memory-efficient, and mathematically rigorous data engineering pipeline designed to prepare, clean, align, and validate network intrusion datasets for advanced representational learning architectures.

Building directly upon the empirical findings of Stage 1, this stage addresses cross-corpus structural mismatches, eliminates confounding topology artifacts, standardizes threat labels, and leverages GPU acceleration for high-performance feature scaling.

---

## 📂 Directory Structure

```
stage2/
├── artifacts/                  # Persisted data transformers, scalers, and pipeline objects
├── figures/                    # Preprocessing validation plots and memory profiling charts
├── logs/                       # Detailed execution logs from preprocessing scripts
├── reports/                    # Generated validation reports (e.g., validation_report.md)
├── scripts/                    # Sequentially numbered data engineering scripts
├── tables/                     # Processed tabular summaries and export logs
└── README.md                   # Stage 2 Documentation (This File)
```

---

## 📜 Executable Script Pipeline

The `scripts/` directory contains 11 dedicated Python modules that execute the full Stage 2 data engineering and validation workflow:

| Script Name | Description & Objective |
| :--- | :--- |
| `01_feature_selection.py` | Selects robust, invariant flow features while dropping confounding topology artifacts. |
| `02_invalid_values.py` | Systematically detects and replaces invalid values (e.g., infinite flow byte calculations). |
| `03_missing_values.py` | Imputes or removes missing values and NaNs across divergent dataset structures. |
| `04_attack_family_mapping.py` | Maps disparate, dataset-specific attack classes into standardized threat families. |
| `05_binary_labels.py` | Derives unified binary classification labels (`BENIGN` vs. `ATTACK`) across all corpora. |
| `06_feature_alignment.py` | Mathematically aligns and reconciles diverging feature names across 2017, 2018, and Lycos. |
| `07_scaler_experiment.py` | Evaluates feature scaling strategies (Standard, MinMax, Robust, Quantile) for non-linear distributions. |
| `08_gpu_validation.py` | Validates GPU-accelerated tensor operations and fast data transformations. |
| `09_dataset_export.py` | Exports fully cleaned, aligned, and optimized datasets into highly efficient storage formats. |
| `10_pipeline_validation.py` | Conducts end-to-end integrity checks on the complete preprocessing workflow. |
| `11_memory_analysis.py` | Profiles memory consumption, optimizing data types to minimize RAM/VRAM footprints. |

---

## 🚀 Key Technical Highlights

- **Standardized Feature Space:** Successfully reconciles disparate column naming conventions (`dst_port` vs. `Dst Port`, `fwd_pkt_cnt` vs. `Tot Fwd Pkts`) to allow seamless cross-dataset model transfer.
- **Robust Outlier & Inf Handling:** Prevents gradient explosions during downstream neural network training by systematically capping or replacing infinite values generated during packet capture anomalies.
- **Advanced Scaler Evaluation:** Compares linear and non-linear scaling techniques to preserve minority attack class distributions without distorting long-tail benign traffic features.
- **GPU Acceleration & Memory Optimization:** Significantly reduces memory overhead through precise data type downcasting (e.g., `float64` to `float32`/`float16`) and leverages GPU acceleration for massive dataset transformations.

For detailed validation summaries and GPU benchmarks, refer to [`reports/validation_report.md`](reports/validation_report.md) and [`reports/gpu_report.txt`](reports/gpu_report.txt).
