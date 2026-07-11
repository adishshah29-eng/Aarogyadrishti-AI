# Developer Handoff Documentation: Chaining & Comorbidity Risk Index (CRI) Engine

This document provides detailed integration guidelines and code snippets for using the comorbidity chaining mechanism and the Comorbidity Risk Index (CRI) calculation inside the Streamlit dashboard.

---

## 1. Exported Python Modules
The CRI calculations and full patient risk profiling interfaces are defined in:
* **CRI and Patient Profile**: [src/chaining/cri.py](file:///c:/Users/Sayli/OneDrive/Desktop/Aarogyadrishti-AI/src/chaining/cri.py)
* **Chaining Engine Evaluation**: [src/chaining/chain_engine.py](file:///c:/Users/Sayli/OneDrive/Desktop/Aarogyadrishti-AI/src/chaining/chain_engine.py)

---

## 2. Risk Profiling Signatures

### Single Patient Full Risk Profile
This function calculates the risk scores for all four diseases (Diabetes, CKD, Hypertension, and Heart Disease) along with the Comorbidity Risk Index (CRI). It automatically chains predictions under the hood (upstream predictions feed downstream models).

```python
def get_full_risk_profile(patient_df) -> dict:
```
* **Inputs**: Can be a python `dict`, a pandas `Series`, or a single-row `DataFrame` containing patient checkup details (canonical schema format).
* **Outputs**: A dictionary containing float risk scores (0.0 to 1.0) and the comorbidity index:
  ```python
  {
      'diabetes_risk': float,
      'ckd_risk': float,
      'hypertension_risk': float,
      'heart_risk': float,
      'cri': float
  }
  ```

---

## 3. Comorbidity Risk Index (CRI) Calculation Details

The CRI is calculated using `compute_cri(diabetes_risk, ckd_risk, heart_risk, hypertension_risk) -> float`.

### Math Formula & Rationale
The index is designed to reflect clinical odds ratios where co-occurring metabolic syndrome components amplify cardiac risk non-linearly:

$$CRI = 0.30 \times P(\text{Diabetes}) + 0.20 \times P(\text{CKD}) + 0.25 \times P(\text{Heart Disease}) + 0.25 \times P(\text{Hypertension}) + \text{Interactions}$$

Where the interactions are defined as:
* **Diabetes + Heart Disease**: $+0.15 \times P(\text{Diabetes}) \times P(\text{Heart Disease})$ (reflects that diabetics are 2–4× more likely to develop coronary diseases).
* **Hypertension + Heart Disease**: $+0.10 \times P(\text{Hypertension}) \times P(\text{Heart Disease})$ (vascular arterial strain multiplier).
* **Diabetes + Hypertension**: $+0.05 \times P(\text{Diabetes}) \times P(\text{Hypertension})$ (overlapping metabolic syndrome strain).

*Note: The calculated score is clipped to the $[0.0, 1.0]$ range.*

---

## 4. Usage Example

### Example: Running a Full Patient Profile
```python
from src.chaining.cri import get_full_risk_profile

# Patient checkup input matching canonical schema
patient_checkup = {
    'patient_id': 'pt_202',
    'age': 58.0,
    'sex': 1.0,            # Male
    'bmi': 29.5,
    'systolic_bp': 142.0,  # Stage 2 Hypertension range
    'diastolic_bp': 88.0,
    'glucose': 156.0,      # Elevated glucose
    'cholesterol': 240.0,  # High cholesterol
    'smoking': 1.0,        # Active smoker
    'alcohol': 0.0,
    'physical_activity': 0.0,
    'family_history': 0.65 # Pedigree score
}

# Run the full risk profile
profile = get_full_risk_profile(patient_checkup)

print("--- Patient Risk Profile ---")
print(f"Diabetes Risk:     {profile['diabetes_risk']:.2%}")
print(f"CKD Risk:          {profile['ckd_risk']:.2%}")
print(f"Hypertension Risk: {profile['hypertension_risk']:.2%}")
print(f"Heart Disease Risk:{profile['heart_risk']:.2%}")
print(f"Comorbidity Index (CRI): {profile['cri']:.2%}")
```
