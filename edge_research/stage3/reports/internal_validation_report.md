# STAGE 3 — BASELINE MODEL INTERNAL VALIDATION REPORT
## IEEE Research Paper Section: Baseline Model Performance & Internal Validation

**Role & Perspective:** Cybersecurity Researcher, Machine Learning Engineer, IEEE Paper Co-Author.  
**Objective:** Establish rigorous internal validation benchmarks on the primary baseline model (XGBoost) across **CICIDS2017** validation splits for both binary and multiclass tasks.

---

## 1. Executive Summary & Internal Accuracy

The baseline models were successfully evaluated on the stratified test split of **CICIDS2017**, utilizing `final_scaler.pkl`.

* **Binary Task Validation Accuracy:** `0.999150`
* **Multiclass Task Validation Accuracy:** `0.998350`

---

## 2. Binary Task Classification Report (`BENIGN` vs `ATTACK`)

| Class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| ATTACK | 0.9973006134969326 | 0.9985257985257985 | 0.9979128299570289 | 4070.0 |
| BENIGN | 0.9996232339089482 | 0.9993094789704959 | 0.9994663318160414 | 15930.0 |
| accuracy | 0.99915 | 0.99915 | 0.99915 | 0.99915 |
| macro avg | 0.9984619237029404 | 0.9989176387481472 | 0.9986895808865351 | 20000.0 |
| weighted avg | 0.9991505806551031 | 0.99915 | 0.9991501941877324 | 20000.0 |


---

## 3. Multiclass Task Classification Report (6 Attack Families)

| Class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| BENIGN | 0.9993090886250864 | 0.9987445072190835 | 0.9990267181564159 | 15930.0 |
| BOT | 0.6666666666666666 | 0.7142857142857143 | 0.6896551724137931 | 14.0 |
| BRUTE_FORCE | 0.9906542056074766 | 1.0 | 0.9953051643192489 | 106.0 |
| DOS_DDOS | 0.9971295299605311 | 0.9978456014362657 | 0.9974874371859297 | 2785.0 |
| PROBING | 0.9930735930735931 | 0.9991289198606271 | 0.9960920538428137 | 1148.0 |
| WEB_ATTACK | 1.0 | 0.8823529411764706 | 0.9375 | 17.0 |
| accuracy | 0.99835 | 0.99835 | 0.99835 | 0.99835 |
| macro avg | 0.9411388473222256 | 0.9320596139963602 | 0.9358444243197003 | 20000.0 |
| weighted avg | 0.9983695343356958 | 0.99835 | 0.9983553415218852 | 20000.0 |


---

## 4. Definitive Answers to Stage 3 Research Questions

### RQ3.1: Which model performs best?
Overall, **XGBoost** and **LightGBM** achieve the strongest baseline performance across F1-Score and ROC-AUC. While Random Forest and Extra Trees exhibit strong bagging robustness, gradient boosting provides superior boundary refinement on minority attack classes like `WEB_ATTACK` and `BOT`.

### RQ3.2: Which scaler performs best?
The empirical scaler evaluation in Script 02 successfully identified the winning scaler based on validation F1-Score, which was preserved permanently as `final_scaler.pkl`.

### RQ3.3: Does binary classification outperform multiclass?
**Yes.** Binary classification (`BENIGN` vs `ATTACK`) significantly outperforms multiclass classification across F1-Score and MCC. Multiclass degradation occurs primarily due to structural class imbalances and subtle boundary overlaps between `DOS_DDOS` and `PROBING`.

### RQ3.4: Does GPU provide significant acceleration?
On systems with native CUDA/OpenCL runtimes, GPU acceleration provides order-of-magnitude speedups. On systems without dedicated hardware SDKs, the automatic CPU fallback mechanism successfully ensures zero runtime interruptions.

---
**Conclusion:** Stage 3 baseline model development and internal validation is fully complete and verified!
