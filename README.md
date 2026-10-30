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
- **Evaluation**: Headline metrics are 5-fold cross-validated. ROC and calibration curves are computed on a genuinely **held-out 20% test split** — a fresh model with the shipped configuration is trained on the 80% train split and scored on the untouched 20% (no in-sample leakage). See `scripts/generate_eval_cache.py` and `scripts/paper_artifacts.py`.
- **Lab-report reading**: An optional upload on wizard Step 1. A PDF, photo, or Word file is read in memory and used to pre-fill the form. The file is not saved. The consent line does not name the reading service.

### Features
* **Checkup-Safe Inference**: All models use only non-invasive, routine checkup features (BMI, Blood Pressure, Glucose, demographics, lifestyle). No specialized lab panels are required.
* **What-If Simulator**: Clinicians can adjust modifiable factors (e.g., BMI, Glucose) and observe real-time impacts on the Comorbidity Risk Index.
* **Clinical Guardrails**: Built-in input validation prevents physiologically impossible data points (e.g., Systolic BP < Diastolic BP).
* **Research Transparency**: The dashboard includes a dedicated "Model Evidence & Research Validation" panel detailing accuracy, chaining deltas, and ethical limitations.
* **Lab-report pre-fill**: On Step 1, a patient can upload a blood-test PDF, image (PNG/JPG/WEBP), TXT, or DOCX. Accepted files are at most 10 MB. Values are shown for review before they fill the wizard.

---

## 🚀 Setup & Execution

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Paper figures also need Matplotlib, which is not in that file: `pip install matplotlib`.
3. **Lab-report key** (only if you want upload reading). From the repo root, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put the key in quotes:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
   `.streamlit/secrets.toml` is git-ignored. Streamlit reads it only when the app is started from this folder. On Streamlit Community Cloud, paste the same two lines into **Settings → Secrets** — the local file is not uploaded with the repo. Without a key, the sidebar accepts one for that browser session only, and the rest of the wizard still works.
4. **Run the Dashboard** from the repo root:
   ```bash
   python -m streamlit run src/dashboard/app.py
   ```

The reader uses `gemini-3.5-flash` with thinking kept minimal. `gemini-2.5-flash` is no longer available to new keys, and `gemini-3.8-flash` was returning 503s. If `gemini-3.5-flash` is overloaded, the app tries `gemini-3.5-flash-lite` once. Override the model with `GEMINI_MODEL` in `secrets.toml`.

---

## 📄 IEEE paper figures

`scripts/paper_artifacts.py` reuses the held-out split in `scripts/generate_eval_cache.py` (stratified 20%, `random_state=42`, SMOTE on the training split only, shipped features and chaining). It writes `reports/paper/`:

| File | Contents |
|---|---|
| `roc_curves.png` / `.pdf` | All four ROC curves, single-column IEEE size (3.4 × 2.8 in, 300 dpi) |
| `calibration_curves.png` / `.pdf` | Reliability curves, 10 quantile bins, Brier score in the legend |
| `diabetes_thresholds.csv` | Default 0.5, Youden's J, and the highest threshold with sensitivity ≥ 85% |
| `test_set_summary.csv` | Held-out size and prevalence |
| `environment_info.txt` | Library versions, hardware, `model.get_params()`, and the CRI formula |

Held-out AUC / Brier on that split: Diabetes 0.863 / 0.110, Heart 0.688 / 0.168, CKD 0.780 / 0.145, Hypertension 0.811 / 0.172. Held-out sizes: Diabetes 1,583, Heart 848, CKD 1,031, Hypertension 1,628. Five-fold CV for the shipped models is in `reports/baseline_metrics.md` (Diabetes 81.75% acc, AUC 0.8683, F1 0.5284; CKD 81.55%, 0.7943, 0.5290; Heart 75.59%, 0.7121, 0.3528; Hypertension 73.19%, 0.8101, 0.6668).

---

## ⚠️ Limitations & Ethics
AarogyaDrishti AI is a **screening tool, not a diagnostic instrument**. It is intended to triage patients and highlight unseen risk vectors, but **results must be reviewed by a qualified clinician**.
* **Diabetes data**: The diabetes model trains on **NHANES 2021–2023** (7,912 adults, both sexes), not the older synthetic Kaggle set. The label is HbA1c ≥ 6.5% or a doctor diagnosis. Held-out AUC is 0.863; 5-fold CV AUC is 0.868. F1 stays near 0.53 because about 15% of rows are positive.
* **Heart data**: the heart model trains on the **Framingham Heart Study** (4,240 records, both sexes, 10-year CHD). Predicting future CHD from routine features is hard: 5-fold CV AUC is 0.712 and the held-out AUC is 0.688. Accuracy is dominated by the roughly 85% negative class, so F1 stays near 0.35.
* **CKD data**: The CKD model trains on **NHANES 2017–2018** (5,154 adults). CKD is labelled by eGFR < 60 (race-free CKD-EPI 2021) or urine ACR ≥ 30. Checkup-safe AUC is 0.794 on 5-fold CV and 0.780 on the held-out split. F1 is about 0.53 at the default threshold because about 19% of rows are positive.
* **Cross-dataset chaining caveat**: The upstream Diabetes/CKD risk features consumed by the Heart and Hypertension models are, at training time, computed on datasets that lack some of their inputs (median-imputed). Those upstream features therefore carry less signal during training than they do at live inference, where the full checkup panel is present. This is an inherent limitation of chaining models each trained on a different public dataset; it is documented, not a validated end-to-end training pipeline.
* **General bias**: The models are trained on public datasets (NHANES, Framingham) which carry demographic biases. Hypertension uses a doctor-diagnosed label, which is narrower than a measured-BP definition and is partly self-reported at the exam.
