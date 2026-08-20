# External Validation — NHANES August 2021-August 2023

The CKD, Heart Disease, and Hypertension shipped models were tested on a cohort **none of them were trained on**: NHANES August 2021-August 2023, a nationally representative US survey of different people than any of their training datasets (CKD: NHANES 2017-2018; Heart: Framingham; Hypertension: cardio/Kaggle dataset). Ground-truth labels are derived independently of our models, from objective labs and doctor-diagnosis questionnaire answers (doctor-diagnosed hypertension/heart disease, eGFR + urine ACR for CKD) — never from our own predictions, so there is no circularity.

**Diabetes is excluded from this table.** As of the P0 monotonic-constraints fix, the Diabetes model now trains directly on this same NHANES 2021-2023 cohort (replacing the synthetic 100k Kaggle dataset that caused a fasting-glucose=158 mg/dL patient to be scored 4.1% "LOW" risk). Scoring the Diabetes model on this cohort would therefore be in-sample evaluation, not external validation — its honest out-of-sample estimate is the 5-fold CV in `reports/baseline_metrics.md` (Accuracy 80.67%, AUC 0.8475).

## Results

| Disease | N (labelled) | Prevalence | AUC | Accuracy | F1 | Confusion (TN,FP,FN,TP) |
|---|---|---|---|---|---|---|
| CKD | 5658 | 17.2% | 0.7562 | 78.49% | 0.4394 | [3964, 722, 495, 477] |
| Heart Disease | 7772 | 8.0% | 0.7455 | 75.73% | 0.2504 | [5571, 1579, 307, 315] |
| Hypertension | 8139 | 36.3% | 0.7719 | 70.66% | 0.6136 | [3855, 1328, 1060, 1896] |

## Training-time vs. external AUC

| Disease | CV AUC (train dataset) | **External AUC (NHANES 2021-2023, unseen)** | Gap | Verdict |
|---|---|---|---|---|
| CKD | 0.7492 | **0.7562** | +0.007 | ✅ Holds up |
| Heart Disease | 0.6863 | **0.7455** | +0.059 | ✅ Holds up (better on a larger sample) |
| Hypertension | 0.7998 | **0.7719** | −0.028 | ✅ Small, expected drop |

All three externally-validated models generalize well — none shows the kind of large drop that would indicate overfitting to training-set quirks.

## How to read this

- This is a **stricter** test than the 5-fold CV or the 80/20 held-out split reported in `reports/baseline_metrics.md` — those come from the SAME dataset each model was trained on. This cohort is entirely new people, from a different survey wave.
- A model that scores similarly here to its training-time metrics has **genuinely generalized**. A large drop indicates the training metric was optimistic (overfit to that dataset's quirks) or the label definitions do not perfectly match across datasets (see caveats).

## Caveats

- **Label definitions are not identical to each training target.** E.g. our Hypertension model's training target ('cardio') is a broader self-reported cardiovascular-disease flag from the Kaggle dataset, while the NHANES ground truth here is specifically doctor-diagnosed high blood pressure (BPQ020) — the closest available NHANES signal, but not an exact match. Similarly Heart Disease compares against doctor-diagnosed CHD/angina/heart-attack, the closest analogue to Framingham's 10-year CHD outcome (which is prospective, not point-in-time).
- Some features are missing per person (e.g. fasting glucose is only collected on a NHANES sub-sample); each model fills gaps with its own training medians, exactly as it would for a real partial checkup.
- CKD requires serum creatinine to be present (for eGFR); rows without it are excluded from CKD evaluation only.
