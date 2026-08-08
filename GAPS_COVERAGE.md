# Research Gaps → Project Coverage

The source research paper identified **three gaps**. This document maps each one to what the AarogyaDrishti AI project now demonstrably does to address it, with evidence, and states honestly what still remains.

**The three gaps:**
1. **Lacks explainability**, making results hard to interpret
2. **Limited real-world or clinical deployment**
3. **Uses limited datasets and lacks advanced optimization techniques**

Status legend: ✅ Covered · 🟡 Partially covered · ⏳ Open

---

## Scorecard

| # | Gap from the paper | Status | One-line summary |
|---|---|---|---|
| 1 | Lacks explainability | ✅ **Covered** | Per-prediction SHAP + plain-language clinical advice for every disease |
| 2 | Limited real-world / clinical deployment | 🟡 **Substantially addressed** | Deployable app using only routine checkup inputs, with validation & calibration — but not yet clinically trialled |
| 3a | Uses limited datasets | ✅ **Covered** | Biased/tiny sets replaced & unified; diabetes now 100k mixed-sex |
| 3b | Lacks advanced optimization | 🟡 **Partially addressed** | Gradient boosting + SMOTE + chaining in place; systematic hyper-parameter/threshold optimization still to add |

---

## Gap 1 — Lacks explainability ✅ COVERED

This is now a core strength of the project rather than a gap.

**What we built:**
- **Instance-level SHAP** attributions for **every** disease prediction — the model shows exactly which features pushed each patient's risk up or down.
- A **plain-language clinical advice layer**: SHAP outputs are translated into human-readable "factors elevating your risk" / "factors protecting your health" with actionable guidance, driven by an external **YAML config** (`clinical_guidelines.yaml`) so clinical wording is separated from code.
- The **dashboard** surfaces this per patient: a per-disease risk breakdown, personalized insights, a feature-impact view, and a "For Doctors & Researchers" panel with the underlying evidence.
- Verified in practice: e.g. `ckd_risk` correctly appears as a top driver for Heart Disease, with a tailored kidney-risk explanation.

**Evidence:** `src/explainability/shap_engine.py`, `src/explainability/generator.py`, `src/config/clinical_guidelines.yaml`, the "Personalized Insights" section of `src/dashboard/app.py`.

---

## Gap 2 — Limited real-world / clinical deployment 🟡 SUBSTANTIALLY ADDRESSED

The project is designed for deployability and ships as a working application, though true clinical validation remains future work.

**What we did to make it deployable:**
- **"Checkup-safe" model design** — all shipped models use only **routine, non-invasive checkup inputs** (age, sex, BMI, blood pressure, glucose, cholesterol, lifestyle). No expensive/invasive lab panels are required, which is what makes the tool usable in **low-resource / primary-care settings**.
- A **working, interactive Streamlit dashboard** — a real deployable artifact, not just notebooks — that produces a full risk profile and a Comorbidity Risk Index in real time.
- **Clinical guardrails**: physiologic input validation (e.g. plausible ranges, systolic > diastolic), driven by a single canonical schema (`src/schema.py`).
- **Calibration + leakage-free held-out validation** — trustworthy probabilities are a prerequisite for any clinical use.
- **Honest framing**: presented explicitly as a **screening tool, not a diagnostic instrument**, with limitations disclosed.
- A **"What-If" simulator** so a clinician/patient can see how modifiable factors change the risk.

**What still remains (⏳ open):**
- No **prospective/real-patient clinical validation** or trial yet — results are on public retrospective datasets.
- No **EHR / hospital-system integration**.
- Datasets are cross-sectional (single visit), so risk is a snapshot, not a longitudinal prediction.

**Evidence:** `src/dashboard/app.py`, `src/schema.py`, the "checkup-safe" feature sets in each `src/models/*_model.py`, `reports/baseline_metrics.md`.

---

## Gap 3 — Uses limited datasets and lacks advanced optimization

This gap has two halves; we address them to different degrees.

### 3a. Limited datasets ✅ COVERED

- **Diabetes:** replaced the tiny, **female-only Pima** set (768 rows) with the **100,000-row, mixed-sex** Diabetes Prediction dataset (≈48k F / 31k M adults). Checkup-safe AUC rose **0.83 → 0.91**, and the model is now validated across both sexes.
- **Scale:** the hypertension model trains on **70,000** records; multiple public datasets are unified under one **canonical schema** so they interoperate.
- **CKD upgrade prepared:** the one remaining small set (400-row UCI) has a fully-built, unit-tested **NHANES replacement** pipeline (both-sex, real kidney labs via CKD-EPI 2021 eGFR) ready to run — see `scripts/build_ckd_nhanes.py` and `docs/ckd_nhanes_upgrade.md`. *(🟡 pending the data fetch.)*

**Evidence:** `src/data_pipeline.py`, `data/raw/diabetes_100k.csv`, `GAPS`-related metrics in `reports/baseline_metrics.md`.

### 3b. Advanced optimization techniques 🟡 PARTIALLY ADDRESSED

**What is in place already:**
- **Gradient-boosted trees (XGBoost)** — a strong modern classifier — for all four diseases.
- **SMOTE** applied on training folds only, to handle class imbalance without leakage.
- **Stratified 5-fold cross-validation** for reliable metrics.
- A **chaining / feature-propagation architecture** (upstream risks feed downstream models) — a form of stacked modelling — with an ablation quantifying its effect.

**What is still missing (⏳ open):**
- **Systematic hyper-parameter optimization** — current XGBoost settings are hand-chosen (`n_estimators=100, max_depth=4, lr=0.1`), not searched via Optuna / Bayesian / grid search.
- **Decision-threshold tuning** — predictions use the default 0.5 cut-point; for the imbalanced diabetes model this depresses F1 (a tuned operating point via Youden's J would improve sensitivity for a screening tool).
- **Feature selection / model search** beyond the fixed checkup-safe lists.

This is the least-closed part of the three gaps and is the clearest next piece of work.

**Evidence:** `XGB_PARAMS` in `src/models/heart_model.py` / `hypertension_model.py`, `scripts/retrain_all.py`.

---

## Summary

| Paper gap | Verdict |
|---|---|
| **1. Explainability** | ✅ **Fully addressed** — SHAP + plain-language advice is a project strength. |
| **2. Real-world deployment** | 🟡 **Substantially addressed** — deployable checkup-only app with validation & guardrails; clinical trialling remains future work. |
| **3. Limited datasets & optimization** | Datasets ✅ **addressed** (100k mixed-sex diabetes, 70k hypertension, NHANES CKD prepared); advanced optimization 🟡 **partial** — HPO & threshold tuning still to add. |

**In one line for a review panel:** *Two of the paper's three gaps (explainability, and limited datasets) are directly and demonstrably closed; the third — deployment and advanced optimization — is substantially advanced through a deployable checkup-only app, but full clinical validation and systematic hyper-parameter optimization are the honest next steps.*

---

## Suggested next steps to fully close Gap 3b
1. Add **Optuna hyper-parameter search** per model into `retrain_all.py` (bounded search over depth/estimators/learning-rate/regularization), reporting the tuned CV score.
2. Add **threshold selection** (Youden's J or a target-sensitivity operating point) and report **sensitivity/specificity**, not just accuracy.
3. Complete the **NHANES CKD** data upgrade (already engineered) to finish closing Gap 3a.
