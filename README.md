# AarogyaDrishti AI 🫀

**AarogyaDrishti AI** is an AI-powered clinical screening tool designed to predict comorbidity risk using routine checkup data. It chains multiple independent machine learning models to synthesize a single **Comorbidity Risk Index (CRI)**, capturing the clinical reality that metabolic and cardiovascular diseases do not occur in isolation.

> **Target Venue Alignment**: This project is structured around the standards of the *Machine Learning for Health (ML4H)* conference, emphasizing robust evaluation (ablation of chaining gains), model calibration, and clinical transparency.

---

## 🔬 Core Innovation: Risk Chaining Architecture

Traditional medical AI predicts diseases in silos (e.g., a Diabetes model, a Heart Disease model). AarogyaDrishti uses a **chained architecture**:
1. **Upstream Models**: Independent XGBoost models predict *Diabetes* and *CKD* from patient vitals.
2. **Feature Propagation**: These upstream probabilities are added as genuine input features to the downstream *Heart Disease* and *Hypertension* models — the shipped `.pkl` models are trained with `diabetes_risk` and `ckd_risk` in their feature set, so the propagation happens at inference time, not merely as a post-hoc weighting.
3. **Synthesis**: The four predictions are combined into a weighted Comorbidity Risk Index (CRI) with clinical interaction terms.

### The Evidence for Chaining
We measure the chaining effect two ways — on the **deployed** models and on two **independent** comorbidity cohorts — and report it honestly:
* **Deployed-model ablation** (5-fold CV, isolated vs. chained checkup-safe): Heart Disease **slightly negative** (−0.38% acc, −0.0045 AUC); Hypertension **~neutral** (+0.05% acc, +0.0001 AUC). On both deployed downstream models the upstream signal is largely already captured by the vitals — chaining is not a reliable win on either.
* **Cross-dataset ablation** (`reports/chaining_results.md`): every experiment moves **well under a percentage point** in either direction on two independent comorbidity cohorts.
* **Bottom line**: chaining is a *modest, targeted* prior that helps most in populations with concurrent metabolic risk — **not** a blanket accuracy improvement. See the caveat below on cross-dataset upstream features.

---

## 🛠️ Tech Stack & Clinical Validation

- **Models**: 4 XGBoost Classifiers (5-fold cross-validated, SMOTE for class imbalance).
- **Explainability**: SHAP (SHapley Additive exPlanations) for instance-level feature attribution.
- **Frontend**: Streamlit with custom "Clinical Light" design system using Plotly.
- **Evaluation**: Headline metrics are 5-fold cross-validated. ROC and Calibration curves are computed on a genuinely **held-out 20% test split** — a fresh model with the shipped configuration is trained on the 80% train split and scored on the untouched 20% (no in-sample leakage). See `scripts/generate_eval_cache.py`.

### Features
* **Checkup-Safe Inference**: All models use only non-invasive, routine checkup features (BMI, Blood Pressure, Glucose, demographics, lifestyle). No specialized lab panels are required.
* **What-If Simulator**: Clinicians can adjust modifiable factors (e.g., BMI, Glucose) and observe real-time impacts on the Comorbidity Risk Index.
* **Clinical Guardrails**: Built-in input validation prevents physiologically impossible data points (e.g., Systolic BP < Diastolic BP).
* **Research Transparency**: The dashboard includes a dedicated "Model Evidence & Research Validation" panel detailing accuracy, chaining deltas, and ethical limitations.

---

## 🚀 Setup & Execution

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Dashboard**
   ```bash
   python -m streamlit run src/dashboard/app.py
   ```

---

## ⚠️ Limitations & Ethics
AarogyaDrishti AI is a **screening tool, not a diagnostic instrument**. It is intended to triage patients and highlight unseen risk vectors, but **results must be reviewed by a qualified clinician**.
* **Diabetes data (improved)**: The diabetes model now trains on the 100k-row [Diabetes Prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (CC0), which contains **both sexes** (≈48k female / 31k male adults after cleaning) — replacing the legacy female-only Pima cohort. `sex` is now a genuine feature. Caveats that remain: the outcome is imbalanced (~11% positive, which lowers F1 at the default threshold although AUC is strong ≈0.91), and the set lacks blood-pressure, cholesterol, and family-history fields (imputed at inference).
* **Heart data (improved)**: the heart model now trains on the **Framingham Heart Study** (4,240 records, both sexes, 10-year CHD), replacing the 1,025-row Kaggle set whose checkup features were *inversely* correlated with the label (producing clinically backwards predictions). Framingham's relationships are correct, so risk now rises with age/BP/cholesterol/smoking. Honest trade-off: the reported AUC falls from an inflated 0.97 to a realistic **~0.68** — the old high number reflected fitting broken labels, not real skill; predicting *future* CHD from routine features is genuinely harder.
* **CKD data (upgraded)**: The CKD model now trains on an **NHANES 2017–2018** cohort of **5,154 adults** (both sexes), replacing the 400-row UCI set. CKD is labelled by clinical criteria — eGFR < 60 (race-free CKD-EPI 2021 from serum creatinine) or urine ACR ≥ 30 (KDIGO markers) — via `scripts/build_ckd_nhanes.py`. Honest trade-off: checkup-safe AUC moves from an optimistic 0.89 (400-row, near-perfect separation) to a realistic **0.74** on 5k+ real patients, with clinically correct relationships (risk rises with age/BP/glucose).
* **Cross-dataset chaining caveat**: The upstream Diabetes/CKD risk features consumed by the Heart and Hypertension models are, at training time, computed on datasets that lack some of their inputs (median-imputed). Those upstream features therefore carry less signal during training than they do at live inference, where the full checkup panel is present. This is an inherent limitation of chaining models each trained on a different public dataset; it is documented, not a validated end-to-end training pipeline.
* **General bias**: The models are trained on public datasets (Diabetes Prediction 100k, BRFSS, UCI, Cleveland/Kaggle) which carry demographic biases and rely partly on self-reported data.
