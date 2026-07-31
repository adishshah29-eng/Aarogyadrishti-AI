# Chained Risk Prediction Results (Isolated vs. Chained Performance Comparison)

This report is an **ablation** on two independent comorbidity datasets (a stroke cohort and a BRFSS metabolic-syndrome cohort). It measures what happens to Heart Disease and Hypertension prediction when the upstream Diabetes and CKD risk probabilities are added as features, versus an otherwise-identical isolated model. It complements the deployed-model ablation in `baseline_metrics.md`.

## Experiment Results

| Experiment / Dataset | Isolated Accuracy | Chained Accuracy | Accuracy Delta | Isolated ROC AUC | Chained ROC AUC | ROC AUC Delta |
|---|---|---|---|---|---|---|
| Heart + Hypertension Dataset (Predict Heart Disease) | 88.38% | 88.22% | **-0.16%** | 0.8324 | 0.8313 | **-0.0011** |
| Heart + Hypertension Dataset (Predict Hypertension) | 85.17% | 85.50% | **+0.33%** | 0.7698 | 0.7676 | **-0.0022** |
| BRFSS Dataset (Predict Heart Disease) | 80.98% | 81.10% | **+0.11%** | 0.7793 | 0.7798 | **+0.0005** |
| BRFSS Dataset (Predict Hypertension) | 70.36% | 70.48% | **+0.12%** | 0.7777 | 0.7777 | **-0.0000** |

## Rationale & Key Takeaways

1. **The effect is small and dataset-dependent, not a blanket improvement.** Across the 4 experiments, chaining helped in 3, was roughly neutral in 0, and slightly hurt in 1. The largest accuracy movement was **+0.33%** (on "Heart + Hypertension Dataset (Predict Hypertension)"); every experiment moved by well under a percentage point in either direction. Chaining should be read as a *modest, targeted* prior for comorbid populations, not as a general accuracy win.
2. **Metabolic Syndrome Overlap**: High blood glucose (diabetes) and impaired filtration (CKD) are pathologically linked to vascular strain and atherosclerotic progression, which is the plausible mechanism for any gain the chained model captures. The near-zero deltas elsewhere show the signal is largely redundant once the isolated features are present.

> **Caveat:** upstream risks here are produced by models trained on *different* datasets, with some inputs median-imputed for these cohorts, so the upstream feature is a partial proxy — see the cross-dataset caveat in `reports/handoff_chaining_cri.md`.
