# External Validation — NHANES August 2021-August 2023

All four shipped models were tested on a cohort **none of them were trained on**: NHANES August 2021-August 2023, a nationally representative US survey of different people than any training dataset (Diabetes: 100k Kaggle set; CKD: NHANES 2017-2018; Heart: Framingham; Hypertension: cardio/Kaggle dataset). Ground-truth labels are derived independently of our models, from objective labs and doctor-diagnosis questionnaire answers (HbA1c, doctor-diagnosed diabetes/hypertension/heart disease, eGFR + urine ACR for CKD) — never from our own predictions, so there is no circularity.

## Results

| Disease | N (labelled) | Prevalence | AUC | Accuracy | F1 | Confusion (TN,FP,FN,TP) |
|---|---|---|---|---|---|---|
| Diabetes | 7912 | 15.5% | 0.7731 | 81.02% | 0.4353 | [5831, 856, 646, 579] |
| CKD | 5658 | 17.2% | 0.7583 | 78.70% | 0.4495 | [3961, 725, 480, 492] |
| Heart Disease | 7772 | 8.0% | 0.7359 | 73.03% | 0.2373 | [5350, 1800, 296, 326] |
| Hypertension | 8139 | 36.3% | 0.7624 | 69.73% | 0.5864 | [3928, 1255, 1209, 1747] |

## Training-time vs. external AUC — the real answer to "is the model good?"

| Disease | CV AUC (train dataset) | Held-out AUC (train dataset, 20% split) | **External AUC (NHANES 2021-2023, unseen)** | Gap (held-out → external) | Verdict |
|---|---|---|---|---|---|
| Diabetes | 0.913 | 0.908 | **0.773** | −0.135 | ⚠️ Real generalization gap — see note below |
| CKD | 0.742 | 0.720 | **0.758** | +0.038 | ✅ Holds up (even slightly better) |
| Heart Disease | 0.677 | 0.655 | **0.736** | +0.081 | ✅ Holds up (better on a much larger sample) |
| Hypertension | 0.802 | 0.799 | **0.762** | −0.037 | ✅ Small, expected drop |

**Three of four models generalize well** — CKD and Heart Disease actually score *higher* externally than on their own held-out split (their original held-out sets were small — 1,031 and 848 rows — so this larger 5.6k/7.8k-row external test is a more stable, and reassuring, estimate). Hypertension drops modestly, consistent with the label-definition mismatch noted below.

**Diabetes is the one real gap.** A 0.91 → 0.77 drop is larger than dataset-mismatch alone typically explains, and is the most honest finding of this whole exercise: the training AUC was likely somewhat optimistic for this population, and/or NHANES's stricter HbA1c/doctor-diagnosis label captures a different (probably more clinically severe) slice of "diabetes" than the training dataset's own label. Either way, **0.77 externally is still solid, usable discrimination for a screening tool — just meaningfully lower than the 0.91 headline number**, and that headline should be reported for what it is: an in-distribution estimate.

## How to read this

- This is a **stricter** test than the 5-fold CV or the 80/20 held-out split reported in `reports/baseline_metrics.md` — those come from the SAME dataset each model was trained on. This cohort is entirely new people, from a different survey wave.
- A model that scores similarly here to its training-time metrics has **genuinely generalized**. A large drop indicates the training metric was optimistic (overfit to that dataset's quirks) or the label definitions do not perfectly match across datasets (see caveats).

## Caveats

- **Label definitions are not identical to each training target.** E.g. our Hypertension model's training target ('cardio') is a broader self-reported cardiovascular-disease flag from the Kaggle dataset, while the NHANES ground truth here is specifically doctor-diagnosed high blood pressure (BPQ020) — the closest available NHANES signal, but not an exact match. Similarly Heart Disease compares against doctor-diagnosed CHD/angina/heart-attack, the closest analogue to Framingham's 10-year CHD outcome (which is prospective, not point-in-time).
- Some features are missing per person (e.g. fasting glucose is only collected on a NHANES sub-sample); each model fills gaps with its own training medians, exactly as it would for a real partial checkup.
- CKD requires serum creatinine to be present (for eGFR); rows without it are excluded from CKD evaluation only.
