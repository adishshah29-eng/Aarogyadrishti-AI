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
* **Deployed-model ablation** (5-fold CV, isolated vs. chained checkup-safe): Heart Disease **+0.39% accuracy / +0.004 AUC**; Hypertension **~neutral** (+0.14% acc, +0.000 AUC). The upstream signal helps Heart Disease but is largely redundant for Hypertension, which already has rich vitals.
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
* **CKD data (upgrade ready)**: The CKD model still uses the small 400-row UCI set (near-perfect separation on the full panel; 79.3% on checkup-safe features). A both-sex NHANES replacement is implemented and unit-tested in `scripts/build_ckd_nhanes.py` (race-free CKD-EPI 2021 eGFR + KDIGO albuminuria labelling); it only needs the public NHANES files, which some sandboxes block via egress policy. See `docs/ckd_nhanes_upgrade.md` for the four-step swap.
* **Cross-dataset chaining caveat**: The upstream Diabetes/CKD risk features consumed by the Heart and Hypertension models are, at training time, computed on datasets that lack some of their inputs (median-imputed). Those upstream features therefore carry less signal during training than they do at live inference, where the full checkup panel is present. This is an inherent limitation of chaining models each trained on a different public dataset; it is documented, not a validated end-to-end training pipeline.
* **General bias**: The models are trained on public datasets (Diabetes Prediction 100k, BRFSS, UCI, Cleveland/Kaggle) which carry demographic biases and rely partly on self-reported data.
