# How AarogyaDrishti AI Works

This document explains the whole system end to end: what data it trains on, how
each model works, how the four models are chained together, how the dashboard
collects input and explains its output, and how the whole pipeline is tested
and validated. It reflects the current state of the `main` branch.

For a narrower "what changed and why" history, see `reports/baseline_metrics.md`
(P0–P3 changelog) and `reports/external_validation.md`. This document is the
"how it all fits together" reference.

---

## 1. What this project is

AarogyaDrishti AI predicts a patient's risk of four conditions — **Diabetes**,
**Chronic Kidney Disease (CKD)**, **Heart Disease**, and **Hypertension** —
from routine checkup data (vitals + a standard blood panel), and combines the
four into a single **Comorbidity Risk Index (CRI)**. It is a screening tool,
not a diagnostic instrument: the dashboard says so explicitly, and every
prediction is meant to prompt a conversation with a clinician, not replace one.

The whole system is built to run on **checkup-safe** inputs — things a
routine visit or a standard blood panel already gives you (age, sex, BMI,
blood pressure, blood sugar, cholesterol, smoking, and a handful of
additional measurements added this year: waist circumference, resting pulse,
uric acid, BUN, triglycerides). No specialized imaging, no genetic testing,
nothing that requires a hospital visit beyond a normal checkup.

---

## 2. Architecture at a glance

```
                    ┌─────────────────┐        ┌─────────────────┐
   patient checkup  │  Diabetes model  │        │    CKD model     │
   data (wizard) ──▶│  (upstream)      │        │   (upstream)     │
                    └────────┬─────────┘        └────────┬─────────┘
                             │  diabetes_risk             │  ckd_risk
                             ▼                            ▼
                    ┌──────────────────────────────────────────┐
                    │     Heart Disease model  /  Hypertension   │
                    │     model (downstream, "chained")           │
                    │  consume diabetes_risk + ckd_risk as        │
                    │  extra input features, on top of their own  │
                    │  checkup-safe features                      │
                    └────────────────────┬─────────────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │   Comorbidity Risk Index (CRI)  │
                          │   weighted sum + interaction     │
                          │   terms (src/chaining/cri.py)    │
                          └───────────────────────────────┘
```

**Upstream models** (Diabetes, CKD) predict directly from checkup data.
**Downstream / chained models** (Heart Disease, Hypertension) predict from
their own checkup data *plus* the upstream models' predicted risk
probabilities — the idea being that a patient's diabetes/CKD risk is itself
informative about their heart/hypertension risk, over and above what the raw
vitals say. This mirrors how real clinical stacking scores work (e.g.
combining QRISK-style component scores), and the codebase measures — not
assumes — whether it actually helps for each model (see §6).

All four models are **XGBoost gradient-boosted trees**, trained with
**monotonic constraints** on their clinically-directional features (age,
BMI, blood pressure, glucose, etc. can never *lower* predicted risk — see
§4), and their hyperparameters are tuned per-model with Optuna (§7).

---

## 3. Data pipeline

### Where the training data comes from

| Model | Source | Cohort size | Label |
|---|---|---|---|
| Diabetes | NHANES 2021–2023 | 7,912 adults | HbA1c ≥ 6.5% **or** doctor-diagnosed diabetes |
| CKD | NHANES 2017–2018 | 5,154 adults | eGFR < 60 (CKD-EPI 2021) **or** urine ACR ≥ 30 (KDIGO) |
| Heart Disease | Framingham Heart Study | 4,240 records | 10-year coronary heart disease outcome |
| Hypertension | NHANES 2021–2023 | 8,139 adults | Doctor-diagnosed high blood pressure (BPQ020) |

NHANES (the CDC's National Health and Nutrition Examination Survey) is a
large, nationally representative U.S. health survey — real clinical
measurements from real people, not a synthetic or scraped dataset. Diabetes
and Hypertension both use the **same** NHANES 2021–2023 cycle; CKD uses the
**2017–2018** cycle (chosen originally for its serum creatinine coverage,
which is what CKD's label is built from).

Each cohort is built by its own script:

- `scripts/build_diabetes_nhanes.py`
- `scripts/build_ckd_nhanes.py`
- `scripts/build_hypertension_nhanes.py`

These read the raw NHANES `.XPT` files from `data/raw/nhanes/<cycle>/`,
merge the relevant modules (demographics, body measures, blood pressure,
biochemistry, smoking questionnaire, etc.) on the participant ID (`SEQN`),
apply the label definition, and write a clean CSV to `data/processed/`.
Missing values in the checkup-safe feature columns are median-imputed at
build time.

Heart Disease is the one model that doesn't use NHANES — it trains on the
public Framingham Heart Study dataset (`data/raw/framingham.csv`), which
already has a long-horizon CHD outcome NHANES doesn't provide.

### Why NHANES, and why this replaced earlier datasets

Early in this project, Diabetes and Hypertension trained on synthetic/proxy
public datasets (a 100k-row synthetic Kaggle diabetes set, and a
self-reported "cardio" Kaggle dataset for hypertension). Both had real
problems:

- The synthetic diabetes dataset's `glucose` column had only **18 discrete
  values** — clear evidence of synthetic generation. This caused a
  **non-monotonic tree artifact**: a patient with fasting glucose = 158
  mg/dL (diagnostic for diabetes) was scored **4.1% risk ("LOW")** by the
  old model. This was the bug that triggered the whole NHANES migration.
- The old hypertension dataset's target was a broad self-reported
  cardiovascular-disease flag, not hypertension specifically, and its
  cholesterol/glucose were ordinal categories (1–3), not real lab values.

Both were replaced with NHANES-trained models with real, high-cardinality
lab values and clinically standard label definitions.

---

## 4. The four models

### 4.1 Diabetes (`src/models/diabetes_model.py`)

**16 checkup-safe features:**
`age, sex, bmi, systolic_bp, diastolic_bp, glucose, cholesterol, smoking,
waist_circumference, resting_pulse, uric_acid, cigs_per_day, triglycerides,
pulse_pressure, age_glucose_interaction, lipid_interaction`

The last three are engineered (derived), not directly collected — see §5.

**Monotonic constraints:** age, BMI, systolic BP, glucose, waist
circumference, resting pulse, uric acid, cigarettes/day, triglycerides,
pulse pressure, age×glucose interaction, and the lipid interaction are all
constrained `+1` — meaning XGBoost is mathematically forbidden from ever
letting a higher value of these *decrease* predicted risk, no matter what
pattern it finds in training data. Sex, diastolic BP, cholesterol, and
smoking are left unconstrained because their relationship to diabetes risk
isn't reliably one-directional.

**Current performance (5-fold CV):** Accuracy 81.8%, ROC AUC 0.868, F1 0.528.

### 4.2 CKD (`src/models/ckd_model.py`)

**16 checkup-safe features:**
`age, sex, bmi, systolic_bp, diastolic_bp, glucose, cholesterol, smoking,
waist_circumference, resting_pulse, uric_acid, bun, triglycerides,
pulse_pressure, bmi_age_interaction, uric_acid_sex_norm`

One important exclusion: **serum creatinine is deliberately left out** of
the checkup-safe feature set, even though it's in the training data. CKD's
label (eGFR) is mathematically derived from serum creatinine, so including
it as a *feature* would be leaking the label into the input — the model
would just be reconstructing its own answer key. It's kept in the data only
to build an illustrative "if you had a full lab panel" baseline comparison
row (AUC 0.826), which is *not* the shipped model. **BUN**, by contrast, is
also a kidney-function marker but is *not* used to derive the label, so
it's safe to include — and it turned out to be CKD's single most
informative feature after age.

**Current performance:** Accuracy 81.6%, ROC AUC 0.794, F1 0.529 (checkup-safe,
shipped). Illustrative full-panel baseline with serum creatinine: AUC 0.826.

### 4.3 Heart Disease (`src/models/heart_model.py`)

**17 chained features:**
`age, sex, bmi, systolic_bp, diastolic_bp, glucose, cholesterol, smoking,
heartRate, cigsPerDay, prevalentHyp, BPMeds, pulse_pressure,
mean_arterial_pressure, diabetes_risk, ckd_risk, comorbidity_risk_interaction`

This is the model where **chaining** (consuming upstream diabetes/CKD risk
as input features) most clearly earns its place architecturally — the
Framingham dataset already had four checkup-relevant columns
(`heartRate`, `cigsPerDay`, `prevalentHyp`, `BPMeds`) that used to sit
unused in the raw data; they're now part of the shipped feature set.
`prevalentHyp` ("ever diagnosed with high blood pressure?") and `BPMeds`
("currently on BP medication?") are the two history questions the wizard
asks specifically for this model.

**Current performance (chained, shipped):** Accuracy 75.6%, ROC AUC 0.712,
F1 0.353. For comparison, the isolated (non-chained) version scores 81.0%
accuracy / 0.698 AUC — chaining trades some accuracy for better
discrimination here (see §6).

### 4.4 Hypertension (`src/models/hypertension_model.py`)

**19 chained features:**
`age, sex, bmi, systolic_bp, diastolic_bp, cholesterol, glucose, smoking,
waist_circumference, resting_pulse, uric_acid, bun, triglycerides,
cigs_per_day, pulse_pressure, mean_arterial_pressure, diabetes_risk,
ckd_risk, comorbidity_risk_interaction`

Two features that exist in Framingham/other public hypertension datasets —
`alcohol` and `physical_activity` — are deliberately **not** part of this
model, or any model in this project. NHANES doesn't have an alcohol-use or
physical-activity module downloaded in this environment, and once
Hypertension moved off the old dataset that did have those fields, no
model consumed them any more. Rather than keep asking users two questions
whose answers would just be discarded, they were removed from the wizard
entirely.

**Current performance (chained, shipped):** Accuracy 73.2%, ROC AUC 0.810,
F1 0.667.

---

## 5. Feature engineering

Shared derived-feature logic lives in `src/models/feature_engineering.py`.
Every function takes a dataframe with the required raw columns already
present and returns it with one new column appended, so training and
live prediction compute the exact same value the exact same way.

**Hand-picked, clinically-motivated features** (the original set):

| Feature | Formula | Rationale |
|---|---|---|
| `pulse_pressure` | systolic − diastolic BP | Independent marker of arterial stiffness |
| `mean_arterial_pressure` | diastolic + pulse_pressure/3 | Average perfusion pressure over a cardiac cycle |
| `age_glucose_interaction` | age × glucose (scaled) | Diabetes risk from a given glucose rises faster with age |
| `bmi_age_interaction` | BMI × age (scaled) | Same idea, for CKD |

**SHAP-guided features** (added by asking the trained models what they
actually rely on, instead of guessing more interaction terms by hand —
see `scripts/analyze_shap_interactions.py`):

| Feature | Formula | Found by | Used in |
|---|---|---|---|
| `lipid_interaction` | cholesterol × triglycerides | Diabetes model's top non-redundant interaction | Diabetes |
| `uric_acid_sex_norm` | uric_acid ÷ sex-specific upper limit (7.2 men / 6.0 women) | CKD model's 2nd-strongest interaction (sex × uric_acid) | CKD |
| `comorbidity_risk_interaction` | diabetes_risk × ckd_risk | Hypertension model's single strongest interaction of any pair | Heart, Hypertension |

`analyze_shap_interactions.py` works by training each model, running
XGBoost's `TreeExplainer.shap_interaction_values` over a sample of the
training data, and ranking every feature pair by mean absolute interaction
magnitude. The top pairs tell you where the model is already finding a
combined effect that neither feature captures alone — `comorbidity_risk_interaction`
is the clearest example: it confirms, as an actual model input, an
amplification effect the CRI formula's own interaction terms had already
assumed mathematically (§6) but that no model had ever been given as a
feature.

Not every SHAP-flagged interaction turned out to help once actually
retrained and tuned with it — Heart Disease and Hypertension both came out
roughly flat (within ±0.001 AUC) after adding `comorbidity_risk_interaction`
and re-tuning. That's reported honestly in `reports/baseline_metrics.md`
rather than only reporting the two features that helped (CKD, Diabetes).

**Chained vs. non-chained engineering** — an important implementation
detail: Heart and Hypertension each have two engineering functions,
`_engineer()` (pulse pressure, MAP — needs only that model's own raw
inputs) and `_engineer_chained()` (the comorbidity interaction — needs
`diabetes_risk`/`ckd_risk` to already be resolved first). Every code path
that builds features for these two models — training, live prediction, the
Optuna tuning script, the SHAP interaction analysis script, the held-out
evaluation script, and the dashboard's own SHAP-explanation code — has to
call both functions in the right order. A bug where several of those paths
called only the first one was found and fixed this session (see the commit
history for `6f59d56`).

---

## 6. Chaining and the Comorbidity Risk Index

### How chaining actually happens

`src/models/upstream.py` defines `add_upstream_risks(df)`, which loads the
already-trained Diabetes and CKD models and appends two columns —
`diabetes_risk` and `ckd_risk` — computed from each row's checkup data.
Heart Disease and Hypertension's training scripts call this before
training, so those two probabilities become ordinary input features
alongside everything else. This means training order matters:
**Diabetes and CKD must be trained (and their `.pkl` files written)
before Heart Disease and Hypertension** — `scripts/retrain_all.py`
enforces this order.

At live prediction time, the dashboard calls
`src/chaining/cri.py::get_full_risk_profile()`, which does the same thing
step by step: predict diabetes_risk and ckd_risk first, append them to the
patient's data, then predict heart_risk and hypertension_risk from the
now-chained feature set.

### Does chaining actually help?

The codebase measures this explicitly rather than assuming it, training an
**isolated** (checkup-safe only) and a **chained** (checkup-safe +
upstream risk) variant of both downstream models every time and reporting
the delta:

| Model | Isolated Acc / AUC | Chained Acc / AUC | Δ Accuracy | Δ AUC |
|---|---|---|---|---|
| Heart Disease | 81.0% / 0.698 | 75.6% / 0.712 | −5.4pp | **+0.014** |
| Hypertension | 73.7% / 0.814 | 73.2% / 0.810 | −0.5pp | −0.003 |

Heart Disease's story: chaining measurably improves ranking ability (AUC)
at a real accuracy cost, because the chained model shifts more borderline
cases past the classification threshold into a positive prediction on a
dataset that's ~85% negative (trading precision for recall). Hypertension's
story is closer to flat — after Hypertension moved to a
doctor-diagnosed-hypertension label on NHANES, its own checkup-safe
features already carry most of the signal, so the upstream risk scores add
little. Both are reported as genuine findings, not smoothed over — this
project consistently treats **AUC as the model-selection metric**, since
accuracy at a fixed 0.5 threshold is dominated by class imbalance on every
one of these four datasets.

### The CRI formula

```
CRI = 0.30 × P(Diabetes) + 0.20 × P(CKD) + 0.25 × P(Heart) + 0.25 × P(Hypertension)
    + 0.15 × P(Diabetes) × P(Heart)       ← diabetics are 2–4× more likely to develop CAD
    + 0.10 × P(Hypertension) × P(Heart)   ← vascular strain amplifier
    + 0.05 × P(Diabetes) × P(Hypertension) ← overlapping metabolic syndrome pathway
```
(clipped to `[0, 1]`, implemented in `src/chaining/cri.py::compute_cri()`)

The base weighted sum treats the four risks as if they were independent
contributors; the three interaction terms add extra weight when two
conditions are *both* elevated, because comorbid disease burden is
clinically worse than the sum of its parts. This formula's interaction
logic is exactly what motivated adding `comorbidity_risk_interaction` as
an actual model feature (§5) — the CRI always assumed diabetes+CKD risk
compounds; now the Heart and Hypertension models can act on that
assumption directly, not just have it applied afterward at the CRI stage.

---

## 7. Hyperparameter tuning

`scripts/tune_hyperparameters.py` runs an **Optuna** search (default 60
trials per model) over XGBoost's capacity/regularization knobs —
`n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`,
`subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` — optimizing
5-fold CV mean ROC AUC, with SMOTE applied on the training fold only
(exactly matching each model's own `train_and_evaluate()` procedure, so a
trial's score is directly comparable to the reported CV numbers).

**What's deliberately never tuned: monotonic constraints.** Those encode a
clinical correctness requirement (glucose can never lower predicted
diabetes risk, full stop), not a performance knob to be searched over.

Results are saved to `configs/tuned_hyperparams.json`. Each model's
`XGB_PARAMS` dict reads this file at import time via
`src/models/tuned_params.py::load_tuned()`, layering any tuned values on
top of that model's hand-picked defaults — so tuning is opt-in and
reproducible: delete the file, or a model's key in it, to fall back to the
defaults.

```
python scripts/tune_hyperparameters.py                       # all 4 models, 60 trials each
python scripts/tune_hyperparameters.py --model diabetes --trials 100
```

Current tuned AUC gains over each model's hand-picked defaults: Diabetes
+0.002, CKD +0.011 (later folded into a larger +0.017 gain once BUN was
added — see `reports/baseline_metrics.md`), Heart Disease +0.014
(largest single tuning gain), Hypertension +0.002.

---

## 8. Explainability (SHAP)

Every prediction on the dashboard comes with a **SHAP** (SHapley Additive
exPlanations) breakdown — which features pushed this specific patient's
risk up or down, and by how much. `src/explainability/shap_engine.py`
wraps XGBoost's built-in `TreeExplainer` to compute this per-patient;
`src/explainability/generator.py` (`InsightGenerator`) turns the raw SHAP
values into the plain-language "Factors elevating your risk" /
"Factors improving your outlook" bullets shown on the results page, using
clinical guideline text from a config file so the clinical phrasing is
kept separate from the UI code.

The dashboard computes SHAP explanations for all four models on every
prediction. Because two of them (Heart, Hypertension) need the chained
interaction feature computed first, the SHAP-prep code in
`src/dashboard/app.py` has to run the same two-stage engineering pipeline
described in §5 before calling `explain_prediction()` — this is exactly
the class of bug mentioned there, now fixed and covered by
`tests/test_wizard_e2e.py`, which drives a real browser through the wizard
and checks the "Personalized Insights" section actually renders.

---

## 9. The dashboard (`src/dashboard/app.py`)

Built with **Streamlit**. The user-facing flow is a **3-step wizard**
rather than one long form:

1. **Demographics** — age, sex, height, weight (BMI computed live)
2. **Checkup Numbers** — blood pressure, glucose, cholesterol, plus the
   optional lab-panel fields (waist circumference, resting pulse, uric
   acid, BUN, triglycerides), each with plain-English help text and
   normal-range context
3. **Lifestyle & History** — smoking status (+ cigarettes/day if current),
   prior hypertension diagnosis, current BP medication
4. **Results** — CRI gauge, per-disease risk cards, SHAP-driven
   "Personalized Insights," a patient input summary table, a "For Doctors
   & Researchers" technical panel (model metrics, ROC/calibration curves,
   chaining ablation table), and a what-if risk simulator

**"I don't know" handling:** every optional lab-panel field (waist, pulse,
uric acid, BUN, triglycerides) has an "I don't know" checkbox. Checking it
disables the field and the corresponding value is left out of the payload
sent to the models entirely — not sent as zero or a guess. Every model's
`predict_risk()` fills a genuinely missing feature with that model's own
training-population median, exactly the way a real partial checkup would
be handled.

**State management:** wizard progress is kept in an explicit `PDATA` dict
in `st.session_state`, written to at the end of each step, rather than
relying on Streamlit's per-widget session state to survive across steps —
that turned out not to be reliable when a widget isn't re-rendered on a
given script rerun (a real bug hit and fixed this session).

**`encode_inputs()`** is the single function that turns the wizard's
collected values into the canonical patient-data dict every model
consumes — it's also where units are normalized (e.g. cholesterol/glucose
always in mg/dL) and where a couple of fields get written under two
different names for historical reasons (`resting_pulse` also becomes
`heartRate`, `cigs_per_day` also becomes `cigsPerDay`, matching the
Framingham dataset's original column naming that the Heart model still
uses internally).

---

## 10. Testing

Three test files under `tests/`, runnable with `python -m pytest tests/`:

- **`test_contracts.py`** — guards the *class* of bug an audit found once:
  input-unit mismatches between the dashboard and a model, features the
  UI collects that no model actually uses ("dead inputs"), the chaining
  wiring genuinely working (not just cosmetically present), and numeric
  correctness of the eGFR and CRI helper functions.
- **`test_regression.py`** — locks in specific historical bugs so they
  can't silently come back: the glucose=158 diabetes-risk bug, generalized
  into a monotonicity sweep across every `+1`-constrained feature in every
  model (including the newest ones — BUN, triglycerides,
  `comorbidity_risk_interaction`); the "engineered feature missing from
  SHAP input" bug; and a check that every model gracefully imputes a
  raw feature that's *entirely absent* from the input (not just
  present-and-empty).
- **`test_wizard_e2e.py`** — drives an actual headless browser
  (Playwright) through the full 3-step wizard to the results page and
  checks for Python tracebacks in the rendered HTML. Auto-skips if
  Playwright isn't installed, so it never blocks a plain test run.
  This exists because both of this session's UI bugs (session-state loss
  advancing past Step 2, and the SHAP `KeyError` on the missing chained
  feature) were only visible by actually clicking through the app — no
  unit-level test would have caught either one.

---

## 11. Validation beyond cross-validation

Two extra layers of evidence beyond the 5-fold CV numbers reported per model:

**Held-out 20% split** (`scripts/generate_eval_cache.py`) — trains a
*fresh* model with the shipped configuration on an 80% split and scores
the untouched 20%, specifically to avoid the leakage that would happen if
you evaluated the shipped model (which is retrained on 100% of its data)
against a slice of its own training set. Produces the ROC and calibration
curves shown in the dashboard's "For Doctors" panel. Output:
`models/eval_cache.pkl`.

**External validation** (`scripts/run_external_validation.py`) — tests
the shipped CKD and Heart Disease models on **NHANES August
2021–August 2023**, a cohort neither model has ever seen, with ground
truth derived independently (eGFR + urine ACR for CKD; doctor-diagnosed
CHD/angina/heart-attack for Heart). Diabetes and Hypertension are
deliberately **excluded** from this script — both now train directly on
this same NHANES cycle, so scoring them here would be in-sample
evaluation dressed up as external validation. Their only honest
out-of-sample estimate is the 5-fold CV number reported for each. Output:
`reports/external_validation.md` and `models/external_eval_cache.pkl`.

Both external AUCs currently exceed their CV numbers slightly (CKD:
0.794 CV → 0.796 external; Heart: 0.712 CV → 0.778 external) — evidence
the models generalize rather than having overfit their own training
distribution.

---

## 12. Reproducing the pipeline from scratch

```bash
pip install -r requirements.txt -r requirements-dev.txt

# 1. Build all four training cohorts from raw NHANES/Framingham data
python scripts/build_diabetes_nhanes.py
python scripts/build_ckd_nhanes.py
python scripts/build_hypertension_nhanes.py
# (heart_clean.csv ships pre-built from data/raw/framingham.csv)

# 2. (Optional) re-run hyperparameter search
python scripts/tune_hyperparameters.py

# 3. Train all four models in dependency order (upstream before downstream)
python scripts/retrain_all.py

# 4. Regenerate held-out validation curves and external validation
python scripts/generate_eval_cache.py
python scripts/run_external_validation.py

# 5. (Optional) SHAP interaction analysis, to look for new engineered features
python scripts/analyze_shap_interactions.py

# 6. Run the test suite
python -m pytest tests/ -v

# 7. Launch the dashboard
streamlit run src/dashboard/app.py
```

---

## 13. Repository map

```
data/
  raw/                    NHANES .XPT files, framingham.csv, legacy Kaggle sets
  processed/              Cleaned per-model training CSVs

src/
  models/
    diabetes_model.py     Upstream model + train/predict interface
    ckd_model.py           "
    heart_model.py        Downstream/chained model + train/predict interface
    hypertension_model.py  "
    feature_engineering.py Shared derived-feature functions
    tuned_params.py        Loads Optuna results on top of each model's defaults
    upstream.py             add_upstream_risks() — the chaining glue
  chaining/
    cri.py                 CRI formula + get_full_risk_profile()
  explainability/
    shap_engine.py          Per-patient SHAP value computation
    generator.py             Turns SHAP values into plain-language insights
  dashboard/
    app.py                  The Streamlit wizard + results UI
  schema.py                 Canonical feature names, units, plausible ranges

scripts/
  build_*_nhanes.py         Cohort builders (one per NHANES-sourced model)
  retrain_all.py             Trains all 4 models in dependency order
  tune_hyperparameters.py    Optuna search
  analyze_shap_interactions.py  Finds real (not guessed) interaction features
  generate_eval_cache.py     Leakage-free 80/20 held-out ROC/calibration curves
  run_external_validation.py Tests shipped models on an unseen NHANES cohort

configs/
  tuned_hyperparams.json     Optuna's output, read by tuned_params.py

tests/
  test_contracts.py          General correctness guarantees
  test_regression.py         Locks in specific historical bug fixes
  test_wizard_e2e.py         Browser-driven UI smoke test

reports/
  baseline_metrics.md        Full per-model metrics history (P0→P3 changelog)
  external_validation.md     Out-of-sample validation results
  model_metrics.json         Machine-readable CV metrics (source for the above)
```
