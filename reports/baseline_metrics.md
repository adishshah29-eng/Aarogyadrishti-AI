# AarogyaDrishti AI - Baseline & Checkup-Safe Models Performance Report

This report documents the performance evaluation of the Diabetes, Chronic Kidney Disease (CKD), Heart Disease, and Hypertension models, comparing the baseline (full-feature) models with the checkup-safe (early-warning) versions.

All models were evaluated using **5-fold cross-validation** with **SMOTE** applied only on training splits to handle class imbalance.

---

## 1. Diabetes Model Performance

The diabetes dataset contains 768 patient records (500 negative, 268 positive).

| Model Version | Accuracy | ROC AUC | F1-Score | Confusion Matrix (TN, FP, FN, TP) |
|---|---|---|---|---|
| **Baseline (Full-feature)** | 74.48% | 0.8227 | 0.6629 | TN: 379, FP: 121, FN: 75, TP: 193 |
| **Checkup-Safe (Ships)** | **76.43%** | **0.8248** | **0.6829** | TN: 389, FP: 111, FN: 70, TP: 198 |

### Tradeoff Analysis & Observations
* **Noise Reduction**: Interestingly, the checkup-safe model performed **better** (+1.95% accuracy and +2.0% F1-score) than the full-feature model.
* **Explanation**: The raw dataset contains features with significant missingness (e.g., `Insulin` has ~49% missing values, and `SkinThickness` has ~30% missing values). When these columns are imputed with the median, it injects noise and causes the classifier to overfit. Removing these low-quality, leakage-prone features allows the model to learn a cleaner decision boundary based on high-quality, checkup-safe indicators: `glucose`, `age`, `bmi`, and `family_history`.

---

## 2. Chronic Kidney Disease (CKD) Model Performance

The CKD dataset contains 400 patient records (150 negative/notckd, 250 positive/ckd).

| Model Version | Accuracy | ROC AUC | F1-Score | Confusion Matrix (TN, FP, FN, TP) |
|---|---|---|---|---|
| **Baseline (Full-feature)** | **97.75%** | **0.9980** | **0.9814** | TN: 145, FP: 5, FN: 4, TP: 246 |
| **Checkup-Safe (Ships)** | 79.00% | 0.8924 | 0.8241 | TN: 117, FP: 33, FN: 51, TP: 199 |

### Tradeoff Analysis & Observations
* **Expected Performance Drop**: As anticipated, the checkup-safe CKD model shows a significant drop in accuracy (-18.75% accuracy and -10.56% ROC AUC) compared to the baseline.
* **Explanation**: The baseline model has access to highly diagnostic clinical markers (e.g., `serum_creatinine`, `haemoglobin`, `packed_cell_volume`, `albumin`, urine specific gravity, etc.). These are direct, biochemical indicators of active kidney pathology and allow near-perfect classification. However, because these lab tests are only ordered during active diagnostic workups (post-symptom or post-diagnosis), they cannot be used in a baseline early-warning checkup screening.
* **Early-Warning Utility**: Despite the performance drop, the checkup-safe model achieves a solid **79.00% accuracy** and **0.8924 ROC AUC** using only basic checkup data (`age`, `diastolic_bp`, `glucose`). This makes it highly valuable for flags before symptoms occur.

---

## 3. Heart Disease Model Performance

The Heart Disease dataset contains 1,025 patient records (499 negative, 526 positive).

| Model Version | Accuracy | ROC AUC | F1-Score | Confusion Matrix (TN, FP, FN, TP) |
|---|---|---|---|---|
| **Baseline (Full-feature)** | **99.51%** | **0.9974** | **0.9952** | TN: 497, FP: 2, FN: 3, TP: 523 |
| **Checkup-Safe (Ships)** | 88.59% | 0.9630 | 0.8858 | TN: 451, FP: 48, FN: 69, TP: 457 |

### Tradeoff Analysis & Observations
* **Expected Tradeoff**: The checkup-safe Heart model experiences a drop in accuracy (-10.92% accuracy) compared to the baseline.
* **Explanation**: The baseline model uses highly specific symptoms and advanced diagnostic indicators: type of chest pain (`cp`), ST segment depression (`oldpeak`), number of major vessels colored (`ca`), and thalassemia type (`thal`). These directly signal current cardiac distress or obstruction. The checkup-safe model is restricted to vitals and demographics (`age`, `sex`, resting systolic blood pressure, and `cholesterol`).
* **Strong Early Screening**: An accuracy of **88.59%** and **0.9630 ROC AUC** on checkup-only features represents a highly powerful screening capability for identifying cardiac risk before advanced chest pain occurs.

---

## 4. Hypertension Model Performance

The Hypertension dataset contains 70,000 patient records (35,021 negative, 34,979 positive).

| Model Version | Accuracy | ROC AUC | F1-Score | Confusion Matrix (TN, FP, FN, TP) |
|---|---|---|---|---|
| **Baseline (Full-feature)** | 73.61% | 0.8027 | 0.7247 | TN: 27201, FP: 7820, FN: 10656, TP: 24323 |
| **Checkup-Safe (Ships)** | **73.63%** | **0.8026** | **0.7248** | TN: 27234, FP: 7787, FN: 10672, TP: 24307 |

### Tradeoff Analysis & Observations
* **Consistent Performance**: The checkup-safe model achieves nearly identical results to the full-feature model (73.63% vs. 73.61% accuracy).
* **Explanation**: The only features dropped in the checkup-safe version are raw `height` and `weight`, which are replaced by their derived comorbidity representation: Body Mass Index (`bmi`). Since BMI encapsulates the relevant risk information of weight relative to height, no predictive information is lost.
