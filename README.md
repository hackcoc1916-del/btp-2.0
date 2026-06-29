# BTP 2.0: Advanced Agentic Adaptation & Representational Learning for Autonomous SOC Defense

[![GitHub release](https://img.shields.io/badge/release-v2.0-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)]()
[![Stage 1: Complete](https://img.shields.io/badge/Stage%201-Data%20Audit%20%26%20Domain%20Shift-success)]()
[![Stage 2: Complete](https://img.shields.io/badge/Stage%202-Feature%20Alignment%20%26%20Preprocessing-success)]()

## 📌 Executive Overview

**BTP 2.0** is an advanced research initiative focused on establishing a scientifically rigorous foundation for cross-dataset network intrusion detection, addressing the critical challenges of domain drift, structural mismatch, and topology artifact leakage in AI-driven Security Operations Centers (AI-SOC).

This repository contains the complete experimental pipelines, empirical evaluations, mathematical shift metrics, and scalable feature processing architectures across three foundational network intrusion corpora:
- **CICIDS2017** (Baseline Training Distribution)
- **CSE-CIC-IDS2018** (External Evaluation Set)
- **Lycos-Unicas-IDS2018** (Large-Scale External Evaluation Set)

---

## 🏗️ Project Architecture & Staged Implementation

The research workflow is divided into dedicated, sequential stages. Each stage is fully self-contained with its own executable scripts, empirical logs, generated tables, figures, and technical reports.

```
BTP 2.0/
├── data/                       # Raw datasets (ignored via .gitignore due to size)
│   ├── CICIDS2017/
│   └── datasets for cross validation/
├── edge_research/
│   ├── stage1/                 # STAGE 1: Data Audit & Domain Shift Analysis
│   │   ├── figures/
│   │   ├── logs/
│   │   ├── reports/
│   │   ├── scripts/
│   │   ├── tables/
│   │   └── README.md           # Stage 1 Detailed Documentation
│   └── stage2/                 # STAGE 2: Preprocessing, Feature Alignment & Validation
│       ├── artifacts/
│       ├── figures/
│       ├── logs/
│       ├── reports/
│       ├── scripts/
│       ├── tables/
│       └── README.md           # Stage 2 Detailed Documentation
├── .gitignore
└── README.md                   # Main Project Overview (This File)
```

---

## 🔬 Stage Summaries: What We Have Done

### [Stage 1: Data Audit & Domain Shift Analysis](edge_research/stage1/README.md)
**Objective:** Establish a purely exploratory, analytical, and scientifically rigorous understanding of cross-dataset structural mismatch, domain drift, and topology artifact leakage.
- **Dataset Inventory & Class Distribution:** Quantified class imbalances and exact disk footprint across 30M+ total flows.
- **Class & Threat Family Overlap:** Mapped common attack categories against unseen, zero-day threats in external datasets (e.g., HOIC, LOIC-UDP, Web Brute Force, SQLi).
- **Mathematical Drift Scoring:** Calculated exact Mean Jensen-Shannon Divergence and Wasserstein distances across 70+ shared features, identifying major covariate shifts in flow duration and byte arrival rates.
- **Topology Leakage Identification:** Proved that ephemeral features like `Source Port` and `Timestamp` act as severe confounding artifacts that degrade cross-dataset model generalization.
- **Manifold Visualization:** Generated 2D PCA projections, t-SNE non-linear manifolds, and 70x70 correlation matrices.

### [Stage 2: Preprocessing, Feature Alignment & GPU Validation](edge_research/stage2/README.md)
**Objective:** Build a robust, memory-efficient, and mathematically rigorous data engineering pipeline to align divergent datasets, handle invalid/missing values, and validate accelerated feature scaling.
- **Cross-Corpus Feature Alignment:** Reconciled divergent feature naming conventions (e.g., `dst_port` vs `Dst Port`) and established a standardized feature space.
- **Robust Data Cleaning:** Implemented systematic handling for invalid values (Infs/NaNs), missing values, and negative flow calculations resulting from packet capture anomalies.
- **Threat Label Normalization:** Created unified binary label mappings and consolidated attack family groupings across disparate threat vectors.
- **Scaler Experiments & GPU Acceleration:** Evaluated multiple feature scaling strategies (Standard, MinMax, Robust, Quantile) and validated high-performance GPU-based pipeline execution.
- **Memory Analysis & Pipeline Validation:** Conducted strict memory footprint optimization and end-to-end pipeline integrity testing to ensure seamless downstream ingestion by representational learning models.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.10+ installed along with the required dependencies in your virtual environment:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scikit-learn
```

### Navigating the Stages
Each stage directory contains a dedicated `scripts/` folder with sequentially numbered Python files (e.g., `01_...`, `02_...`). To execute or review a specific stage's pipeline, refer to its individual `README.md`.

---

## 📜 License & Citation
This project is part of an ongoing IEEE research paper initiative. All source code and automated pipelines are available under the MIT License.
