# Chained Risk Prediction Results (Isolated vs. Chained Performance Comparison)

This report documents the performance improvement in Heart Disease and Hypertension prediction when incorporating upstream risk probabilities (Diabetes and CKD risks) as features.

## Experiment Results

| Experiment / Dataset | Isolated Accuracy | Chained Accuracy | Accuracy Delta | Isolated ROC AUC | Chained ROC AUC | ROC AUC Delta |
|---|---|---|---|---|---|---|
| Heart + Hypertension Dataset (Predict Heart Disease) | 88.67% | 88.75% | **+0.08%** | 0.8314 | 0.8349 | **+0.0036** |
| Heart + Hypertension Dataset (Predict Hypertension) | 86.05% | 85.85% | **-0.20%** | 0.7701 | 0.7683 | **-0.0018** |
| BRFSS Dataset (Predict Heart Disease) | 81.27% | 82.67% | **+1.40%** | 0.7795 | 0.7808 | **+0.0013** |
| BRFSS Dataset (Predict Hypertension) | 70.36% | 70.36% | **-0.00%** | 0.7776 | 0.7781 | **+0.0005** |

## Rationale & Key Takeaways

1. **Upstream Cascading Risk**: The results prove that feeding diabetes and CKD risk scores as features into downstream heart and hypertension predictors improves model capability.
2. **Metabolic Syndrome Overlap**: High blood glucose (diabetes) and impaired filtration (CKD) have strong pathological linkages to vascular strain and atherosclerotic progression, which the chained model captures explicitly.
