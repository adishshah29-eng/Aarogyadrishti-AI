# External Validation — NHANES August 2021-August 2023

The CKD and Heart Disease shipped models were tested on a cohort **neither was trained on**: NHANES August 2021-August 2023, a nationally representative US survey of different people than either training dataset (CKD: NHANES 2017-2018; Heart: Framingham). Ground-truth labels are derived independently of our models, from objective labs and doctor-diagnosis questionnaire answers (doctor-diagnosed heart disease, eGFR + urine ACR for CKD) — never from our own predictions, so there is no circularity.

**Diabetes and Hypertension are excluded from this table.** As of the P0 monotonic-constraints fix and the P1 hypertension dataset swap, both models now train directly on this same NHANES 2021-2023 cohort — Diabetes replacing the synthetic 100k Kaggle dataset that caused a fasting-glucose=158 mg/dL patient to be scored 4.1% "LOW" risk, Hypertension replacing the cardio/Kaggle dataset's broad self-reported cardiovascular-disease label with a real doctor-diagnosed-hypertension one. Scoring either model on this cohort would therefore be in-sample evaluation, not external validation — their honest out-of-sample estimates are the 5-fold CV numbers in `reports/baseline_metrics.md` (Diabetes: Accuracy 81.19%, AUC 0.8624; Hypertension: Accuracy 73.20%, AUC 0.8077).

## Results

| Disease | N (labelled) | Prevalence | AUC | Accuracy | F1 | Confusion (TN,FP,FN,TP) |
|---|---|---|---|---|---|---|
| CKD | 5658 | 17.2% | 0.7964 | 82.57% | 0.5099 | [4159, 527, 459, 513] |
| Heart Disease | 7772 | 8.0% | 0.7775 | 70.72% | 0.2825 | [5048, 2102, 174, 448] |

## How to read this

- This is a **stricter** test than the 5-fold CV or the 80/20 held-out split reported in `reports/baseline_metrics.md` — those come from the SAME dataset each model was trained on. This cohort is entirely new people, from a different survey wave.
- A model that scores similarly here to its training-time metrics has **genuinely generalized**. A large drop indicates the training metric was optimistic (overfit to that dataset's quirks) or the label definitions do not perfectly match across datasets (see caveats).

## Caveats

- **Label definitions are not identical to each training target.** Heart Disease compares against doctor-diagnosed CHD/angina/heart-attack, the closest analogue to Framingham's 10-year CHD outcome (which is prospective, not point-in-time).
- Some features are missing per person (e.g. fasting glucose is only collected on a NHANES sub-sample); each model fills gaps with its own training medians, exactly as it would for a real partial checkup.
- CKD requires serum creatinine to be present (for eGFR); rows without it are excluded from CKD evaluation only.
