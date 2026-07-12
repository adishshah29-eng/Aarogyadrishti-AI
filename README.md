# AarogyaDrishti AI 🫀

**AarogyaDrishti AI** is an AI-powered clinical screening tool designed to predict comorbidity risk using routine checkup data. It chains multiple independent machine learning models to synthesize a single **Comorbidity Risk Index (CRI)**, capturing the clinical reality that metabolic and cardiovascular diseases do not occur in isolation.

> **Target Venue Alignment**: This project is structured around the standards of the *Machine Learning for Health (ML4H)* conference, emphasizing robust evaluation (ablation of chaining gains), model calibration, and clinical transparency.

---

## 🔬 Core Innovation: Risk Chaining Architecture

Traditional medical AI predicts diseases in silos (e.g., a Diabetes model, a Heart Disease model). AarogyaDrishti uses a **chained architecture**:
1. **Upstream Models**: Independent XGBoost models predict *Diabetes* and *CKD* from patient vitals.
2. **Feature Propagation**: These upstream probabilities are fed as input features into the downstream *Heart Disease* and *Hypertension* models.
3. **Synthesis**: The predictions are combined into a weighted Comorbidity Risk Index (CRI).

### The Evidence for Chaining
Our ablation studies (comparing isolated predictions vs. chained predictions) reveal a key finding:
* **Chaining improves accuracy by up to +1.40% on the BRFSS dataset**, where the incidence of overlapping metabolic syndrome is highest. 
* This provides empirical evidence that upstream disease risk acts as a powerful prior for downstream vascular diseases, specifically in populations with complex comorbidity burdens.

---

## 🛠️ Tech Stack & Clinical Validation

- **Models**: 4 XGBoost Classifiers (5-fold cross-validated, SMOTE for class imbalance).
- **Explainability**: SHAP (SHapley Additive exPlanations) for instance-level feature attribution.
- **Frontend**: Streamlit with custom "Clinical Light" design system using Plotly.
- **Evaluation**: ROC and Calibration curves computed on a held-out 20% test split.

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
* **Dataset Bias**: The models were trained on public datasets (PIMA Indians, BRFSS) which carry inherent demographic biases and rely heavily on self-reported data.
* **Feature Reduction**: The CKD model was aggressively pruned to use only checkup-safe features (dropping accuracy from 97.8% to 79.0%) to ensure deployability in low-resource settings. Full lab panels are required for clinical diagnosis.
