# AarogyaDrishti AI — Technical Audit

**Date:** 2026-07-25
**Scope:** Full repository scan — data pipeline, four models, chaining engine, CRI, explainability, dashboard, evaluation scripts, reports, and configuration.
**Method:** Static reading of every source file plus **empirical probing** of the shipped `.pkl` models (feature ranges, prediction sweeps, dead-input tests). Every finding below is reproducible with the command shown.

This audit covers what **remains** after the chaining/leakage/diabetes-dataset fixes already landed on `claude/aarogyadrishti-gaps-review-3f4bhj`. Findings are ranked by severity.

---

## Summary

| Severity | Count | Headline |
|---|---|---|
| 🔴 Critical | 1 | Hypertension model receives cholesterol/glucose on the wrong scale — cholesterol is silently ignored at inference |
| 🟠 High | 3 | `family_history` is a dead input; CKD still on 400 rows; `ckd_risk` has no clinical advice text |
| 🟡 Medium | 6 | Threshold not tuned for imbalance; inconsistent risk tiers; no test suite; stale advice text; unvalidated CRI weights; cross-dataset shift |
| 🟢 Low | 8 | Dead code, dead compute, duplication, provenance/licensing gaps |

**Overall:** the modelling core is now sound (real chaining, leakage-free evaluation, mixed-sex diabetes data). The remaining risk is concentrated at the **inference boundary** — what the dashboard sends into the models versus what those models were trained to expect. Finding #1 is a genuine correctness bug that affects live predictions today.

---

## Resolution status (updated 2026-07-25)

A fix pass addressed the correctness and cleanup findings; the data-/research-dependent ones remain open.

| Finding | Status | How |
|---|---|---|
| C1 hypertension cholesterol/glucose scale | ✅ Fixed | Ordinal 1–3 categories mapped to representative mg/dL in `clean_hypertension()`; model retrained. Cholesterol now spans 0.556→0.853 across the clinical range (was frozen at 0.7415). |
| C1a schema not unit-aware | ✅ Fixed | `src/schema.py` now carries `FEATURE_UNITS`/`FEATURE_RANGES` + `range_warnings()`, and the dashboard validation is driven by it (schema is now live code). |
| H1 `family_history` dead input | ✅ Fixed | Removed from the UI, `encode_inputs`, summary table, and config. |
| H3 `ckd_risk` missing advice | ✅ Fixed | Added `ckd_risk` block to `clinical_guidelines.yaml`; removed the two unused keys. |
| M2 3-tier vs 4-tier | ✅ Fixed | Dashboard now uses the documented 4-tier bands (25/50/75: Low/Medium/High/Critical). |
| M6 stale advice text | ✅ Fixed | Lifestyle targets updated to match the binary UI. |
| M5 no tests | ✅ Fixed | Added `tests/test_contracts.py` (7 contract tests: input-range/scale, no-dead-input, chaining wiring, CRI & eGFR numerics). |
| L1 `schema.py` dead | ✅ Fixed | Now imported and enforced (see C1a). |
| L2 dead `test_import.py`/`templates.py` | ✅ Fixed | Both removed. |
| L3 dead SHAP-waterfall compute | ✅ Fixed | Removed 4 unused explainer calls per prediction. |
| L4 duplicated helper | ✅ Fixed | Inline `get_shap_explanation` removed with L3. |
| L7 unused imports | ✅ Fixed | `numpy`, `make_subplots` removed from the dashboard. |
| **H2 CKD still 400 rows** | ⏳ Open | Blocked on fetching NHANES files (env blocks cdc.gov); builder + guide already in repo. |
| **M1 untuned threshold** | ⏳ Open | Reporting/operating-point work — deferred. |
| **M3 CRI weights unvalidated** | ⏳ Open | Needs a multi-disease cohort to calibrate against. |
| **M4 cross-dataset chaining shift** | ⏳ Open | Inherent; needs a single multi-label cohort. |
| L5 fragile SHAP indexing · L6 unused eval metadata · L8 LICENSE/model card/LFS | ⏳ Open | Minor / housekeeping. |

---

## 🔴 CRITICAL

### C1. Hypertension model: cholesterol & glucose are sent on the wrong scale — cholesterol is completely inert

The hypertension model was trained on the Cardiovascular Disease dataset, where `cholesterol` and `glucose` (`gluc`) are **ordinal categories**:
`1 = normal, 2 = above normal, 3 = well above normal`.

Confirmed in the training data:

```
hypertension_clean.csv →  cholesterol: min=1.00 max=3.00 nuniq=3
                          glucose:     min=1.00 max=3.00 nuniq=3
```

But `encode_inputs()` in `src/dashboard/app.py` sends **raw clinical units** — cholesterol in mg/dL (100–400) and fasting glucose in mg/dL (50–400). Every real-world value is far above the model's maximum training split of 3, so the tree saturates on one branch.

**Measured impact** (identical patient, sweeping cholesterol only):

| Input sent | Hypertension risk |
|---|---|
| 150 mg/dL | 0.7415 |
| 200 mg/dL | 0.7415 |
| 250 mg/dL | 0.7415 |
| 300 mg/dL | 0.7415 |
| 350 mg/dL | 0.7415 |
| **category 1** (normal) | **0.5793** |
| **category 2** (above) | **0.6404** |
| **category 3** (well above) | **0.8380** |

Cholesterol is **frozen** — it makes zero difference to the hypertension score anywhere in the clinical range, while the correct encoding spans a **26-percentage-point** swing. Glucose behaves non-monotonically for the same reason. The baseline is also mis-anchored: every patient is scored as if they were in the most extreme category.

This is the single highest-impact defect remaining: the hypertension score shown to users is systematically wrong, and its SHAP attribution for cholesterol is meaningless.

**Reproduce:**
```bash
python -c "
import pandas as pd, sys; sys.path.insert(0,'.')
from src.models.hypertension_model import predict_risk as ht
from src.models.upstream import add_upstream_risks
base={'age':55.0,'sex':1.0,'bmi':28.0,'systolic_bp':135.0,'diastolic_bp':85.0,'smoking':1.0,'alcohol':0.0,'physical_activity':0.0,'family_history':0.2}
for c in [150,200,250,300,350]:
    print(c, ht(add_upstream_risks(pd.DataFrame([dict(base,cholesterol=float(c),glucose=120.0)]))))"
```

**Recommended fix.** Pick one, and apply it consistently:
- **(a) Preferred — bin at the boundary.** Keep clinical units in the UI (they're what a user has) and convert per-model inside the prediction wrapper: cholesterol `<200 → 1`, `200–239 → 2`, `≥240 → 3`; fasting glucose `<100 → 1`, `100–125 → 2`, `≥126 → 3` (standard ATP III / ADA cut-points). This preserves the UI and fixes the model input.
- **(b) Retrain** the hypertension model on a dataset carrying continuous lipids/glucose.

Whichever is chosen, add a **range assertion** in each `predict_risk()` that warns when an input falls outside its training min/max — this class of bug should never be silent again.

### C1a. Root cause: the canonical schema is not unit-aware

`src/schema.py` declares shared column names (`glucose`, `cholesterol`, …) but **no units or admissible ranges**, and different source datasets use the same name for different quantities (mg/dL in heart/diabetes/CKD, ordinal 1–3 in hypertension). The schema is also **never imported anywhere** (see L1), so nothing enforces it. Until the schema carries units + ranges per feature, this class of defect can recur with any new dataset.

---

## 🟠 HIGH

### H1. `family_history` is collected in the UI but used by **zero** models

The sidebar asks "Family History of Chronic Illness" and the input-summary table displays it, but no shipped model consumes it. This is a **regression introduced by the diabetes dataset swap** — the old Pima model used `DiabetesPedigreeFunction`; the replacement 100k dataset has no family-history column, so the feature silently disappeared from the only model that used it.

Verified — toggling it changes nothing:
```
fam_hist=No : diabetes 0.3573 | ckd 0.2807 | ht 0.7147 | heart 0.9490 | cri 0.7107
fam_hist=Yes: diabetes 0.3573 | ckd 0.2807 | ht 0.7147 | heart 0.9490 | cri 0.7107
ANY DIFFERENCE: False
```
Asking a user for information that cannot affect the result is a transparency problem, not just dead code.

**Fix:** either remove the input, or label it clearly as "recorded for the report, not used by the current models". Long term, restore a genuine family-history feature via a dataset that carries one.

### H2. CKD remains the weakest model — 400 rows, 3 features, optimistic baseline

Unchanged by recent work. The shipped CKD model uses only `['age', 'diastolic_bp', 'glucose']` on **400 rows**, while its full-panel baseline reports 97.75% accuracy / 0.998 AUC — near-perfect separation that is a small-sample artefact, not real generalization. CKD risk also feeds the chain as an upstream feature, so its weakness propagates into Heart and Hypertension.

**Status:** the replacement is already built and unit-tested — `scripts/build_ckd_nhanes.py` (race-free CKD-EPI 2021 eGFR + KDIGO albuminuria labelling) with the swap documented in `docs/ckd_nhanes_upgrade.md`. It is **blocked only on fetching the public NHANES files** (this environment's egress policy denies `cdc.gov`). This is the highest-value outstanding data task.

### H3. `ckd_risk` has no entry in `clinical_guidelines.yaml`

Coverage check of model features vs the advice config:
```
MISSING from YAML : ['ckd_risk']
unused in YAML    : ['family_history', 'hypertension_risk']
```
`ckd_risk` is now a **top-3 SHAP driver for Heart Disease** (measured earlier at −0.483 on a test patient), yet the insight generator falls back to the generic string *"This factor is contributing to your overall risk profile."* The patient-facing explanation for one of the strongest drivers is therefore uninformative. `hypertension_risk` is configured but never used as a feature, and `family_history` is stale per H1.

**Fix:** add a `ckd_risk` block mirroring `diabetes_risk`; remove or park the unused keys.

---

## 🟡 MEDIUM

### M1. Diabetes decision threshold is untuned for a ~11% positive rate
CV accuracy 88.8% and AUC 0.913 are strong, but **F1 is 0.553** — the default 0.5 cut-point is poor for an imbalanced screening task, where recall matters more than precision. No threshold selection (Youden's J, or a target-sensitivity operating point) is performed anywhere. For a screening tool, missing true positives is the costlier error. Report sensitivity/specificity at a chosen operating point, not just accuracy.

### M2. Risk tiers in the dashboard contradict the documented framework
`_risk_props()` uses a 3-tier split at **30 / 60** (Low / Moderate / High), while the project's own EDA framework documents a **4-tier** scheme at **25 / 50 / 75** (Low / Medium / High / Critical). Two different risk vocabularies now coexist. Reconcile them, and note that the CRI is a weighted composite whose distribution differs from a raw disease probability — its tier boundaries should be justified on the CRI's own distribution rather than inherited.

### M3. CRI weights and interaction terms are unvalidated
`compute_cri()` uses hand-picked weights (0.30/0.20/0.25/0.25) and interaction coefficients (0.15/0.10/0.05) justified by citation to clinical literature. `run_cri_validation()` only re-checks the arithmetic against hard-coded expected values — it confirms the formula computes what it says, **not that the weights predict anything**. There is no calibration of CRI against an observed comorbidity outcome. Present CRI as a transparent heuristic index (which it is), or validate it on a cohort with multi-disease labels.

### M4. Cross-dataset chaining distribution shift (documented, unresolved)
Upstream `diabetes_risk`/`ckd_risk` are computed at training time on datasets missing several inputs (median-imputed), so those features carry **less signal during training than at live inference**, where the full checkup panel exists. This is disclosed in the README/dashboard, and it plausibly caps the measured chaining gain (+0.39% heart, +0.14% hypertension). A single multi-disease cohort (NHANES, or the 100k set's own `hypertension`/`heart_disease` labels) would let the chain be trained and evaluated end-to-end.

### M5. No automated test suite or CI
There is no `tests/` directory and no CI workflow. Existing checks are ad-hoc scripts (`cri.py` asserts, `build_ckd_nhanes.py --selftest`). Given that finding C1 is exactly the kind of defect a contract test would catch, priority tests are: per-model input-range contracts, a chained-feature presence test, a "no dead input" test, and the eGFR/CRI numerics.

### M6. Advice text in `clinical_guidelines.yaml` is stale versus the binary UI
The config still advises `smoking → "target: Never"`, `alcohol → "target: None or Occasional"`, `physical_activity → "target: Moderate or High"`. The UI was changed to binary (`Non-smoker/Smoker`, `No/Yes`, `Active/Inactive`) to match the models' training encoding, so the guidance references options the user can no longer select.

---

## 🟢 LOW / HOUSEKEEPING

- **L1. `src/schema.py` is dead code** — `CANONICAL_FEATURES` is defined and never imported anywhere. Either enforce it in the pipeline (ideally extended with units/ranges per C1a) or delete it.
- **L2. `src/dashboard/test_import.py` is a scratch debug script** — it prints `dir(templates)` to diagnose a missing function. It is not a test. It is also the **only** reference to `src/explainability/templates.py`, making that module dead too. Remove both, or convert into a real test.
- **L3. Dead compute in the dashboard** — `shap_waterfalls` runs **four `shap.Explainer` calls** per submission (lines ~935–938) and the result is **never rendered**. This is pure latency on every prediction. Similarly `make_shap_bar()` and `make_whatif_gauge()` are defined but never called.
- **L4. Duplicated helper** — `get_shap_explanation()` is defined both inside `app.py` (line 920) and in `shap_engine.py` (line 31). Import the shared one.
- **L5. Fragile SHAP compatibility** — `explain_prediction()` indexes `shap_values[1][0]` for the list case, a legacy binary-classification format. Pin behaviour or branch on the SHAP API version explicitly; a SHAP upgrade could silently flip the class whose attributions are shown.
- **L6. Cached evaluation metadata is unused** — `eval_cache.pkl` stores `n_test` and `pos_rate` per model, but the dashboard never displays them. Showing test-set size and prevalence would materially strengthen the "For Doctors & Researchers" panel (e.g. CKD's held-out AUC of 0.925 rests on just **80** test rows).
- **L7. Unused imports in `app.py`** — `numpy as np` and `make_subplots`.
- **L8. Repository hygiene** — `data/` is ~76 MB committed directly to git (`.git` is 16 MB packed); consider Git LFS or DVC for the raw datasets. There is also **no LICENSE file** and **no model card / dataset provenance document** recording each dataset's source URL, license, collection period, and known biases — which a health-ML project should have, and which reviewers of an ML4H-style submission will look for.

---

## Recommended order of work

1. **Fix C1** (hypertension cholesterol/glucose binning) + add input-range warnings — correctness bug affecting live output.
2. **Fix H1 and H3** — remove or relabel the dead `family_history` input; add `ckd_risk` advice. Both are small and user-visible.
3. **Land H2** — fetch the NHANES files and run the prepared CKD upgrade.
4. **M1** — choose and report a proper operating threshold for the imbalanced diabetes model.
5. **M5** — add a small test suite covering the contracts above, so C1-class regressions fail loudly.
6. **M2, M6, L1–L8** — consistency and cleanup.

## What is already solid (for balance)

- Chaining is genuinely wired end-to-end: downstream models train on and consume `diabetes_risk`/`ckd_risk` (verified — `ckd_risk` appears as a top SHAP driver for Heart Disease).
- Evaluation is leakage-free: held-out curves train a fresh model on the 80% split and score the untouched 20%.
- The diabetes model is now mixed-sex on 79,444 adults (AUC 0.83 → 0.91), with `sex` a live feature (male 0.466 vs female 0.340 on otherwise identical inputs).
- 5-fold CV with SMOTE applied **only** to training folds — methodologically correct.
- Reported chaining gains are honest and modest, matching the measured ablations rather than overstating them.
- Clean separation of concerns: SHAP engine, YAML-driven clinical advice, and UI are properly decoupled.
- Reproducible regeneration path: `retrain_all.py` → `generate_eval_cache.py` → `chain_engine.py`, with metrics emitted to `reports/model_metrics.json`.
