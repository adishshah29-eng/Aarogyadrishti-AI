# Developer Handoff Documentation: Diabetes & CKD Predictors

This document provides function signatures, expected input shapes, and code examples for importing and running the Diabetes and Chronic Kidney Disease (CKD) risk predictors in the Streamlit dashboard.

---

## 1. Exported Python Modules
The prediction functions are defined in:
* **Diabetes**: [src/models/diabetes_model.py](file:///c:/Users/Sayli/OneDrive/Desktop/Aarogyadrishti-AI/src/models/diabetes_model.py)
* **CKD**: [src/models/ckd_model.py](file:///c:/Users/Sayli/OneDrive/Desktop/Aarogyadrishti-AI/src/models/ckd_model.py)

The saved models are located in `models/diabetes_model.pkl` and `models/ckd_model.pkl`.

---

## 2. Prediction Signatures

### Single Patient Risk Prediction
Used to get the 0-1 risk score for a single patient.

```python
def predict_risk(patient_df) -> float:
```
* **Inputs**: Can be a python `dict`, a pandas `Series`, or a single-row `DataFrame` containing patient checkup data.
* **Outputs**: A float between `0.0` and `1.0` representing the probability of risk.

### Batch Patient Risk Prediction
Used to run predictions on multiple patients.

```python
def predict_risk_batch(patient_df) -> pd.DataFrame:
```
* **Inputs**: A pandas `DataFrame` of patient checkup records.
* **Outputs**: A pandas `DataFrame` with columns:
  * `patient_id` (matches input patient IDs, or is an integer index if not provided)
  * `diabetes_risk` (for diabetes model) OR `ckd_risk` (for CKD model) containing float probabilities.

---

## 3. Expected Input Feature Format (Canonical Schema)

Input dataframes or dicts should conform to the **canonical feature schema** (other unused schema fields will be automatically ignored or imputed with training medians, so you do not need to clean the input data beforehand).

The full canonical schema columns are:
* `patient_id` (string)
* `age` (float, years)
* `sex` (float, `1.0` = Male, `0.0` = Female)
* `bmi` (float, kg/m²)
* `systolic_bp` (float, mmHg)
* `diastolic_bp` (float, mmHg)
* `glucose` (float, mg/dL)
* `cholesterol` (float, mg/dL or category)
* `smoking` (float, binary `1.0`/`0.0`)
* `alcohol` (float, binary `1.0`/`0.0`)
* `physical_activity` (float, binary `1.0`/`0.0`)
* `family_history` (float, pedigree score or family history flag)

---

## 4. Usage Examples

### Example 1: Single Patient Prediction (using dict)
```python
import pandas as pd
from src.models.diabetes_model import predict_risk as predict_diabetes
from src.models.ckd_model import predict_risk as predict_ckd

# Input data from dashboard fields
patient_data = {
    'patient_id': 'pt_101',
    'age': 45.0,
    'sex': 0.0,            # Female
    'bmi': 28.4,
    'diastolic_bp': 82.0,
    'glucose': 145.0,
    'family_history': 0.55 # pedigree function for diabetes
}

# Run predictions
p_diabetes = predict_diabetes(patient_data)
p_ckd = predict_ckd(patient_data)

print(f"Diabetes Risk Probability: {p_diabetes:.2%}")
print(f"CKD Risk Probability:      {p_ckd:.2%}")
```

### Example 2: Batch Prediction (using DataFrame)
```python
import pandas as pd
from src.models.diabetes_model import predict_risk_batch as predict_diabetes_batch
from src.models.ckd_model import predict_risk_batch as predict_ckd_batch

# Load dataset
batch_df = pd.DataFrame([
    {'patient_id': 'A01', 'age': 50.0, 'sex': 1.0, 'bmi': 31.2, 'diastolic_bp': 90.0, 'glucose': 180.0, 'family_history': 0.62},
    {'patient_id': 'A02', 'age': 32.0, 'sex': 0.0, 'bmi': 22.0, 'diastolic_bp': 70.0, 'glucose': 95.0,  'family_history': 0.15}
])

# Run batch predictions
diabetes_results = predict_diabetes_batch(batch_df)
ckd_results = predict_ckd_batch(batch_df)

print("--- Diabetes Risk Output ---")
print(diabetes_results)

print("\n--- CKD Risk Output ---")
print(ckd_results)
```

---

## 5. Robustness & Imputation Guard
If the input data contains missing values (`NaN`) or if some columns are omitted entirely, the prediction wrappers will automatically impute them with the training medians under the hood:
* **Diabetes Medians used**: `{'age': 33.0, 'sex': 0.0, 'bmi': 32.0, 'diastolic_bp': 72.0, 'glucose': 117.0, 'family_history': 0.3725}`
* **CKD Medians used**: `{'age': 55.0, 'diastolic_bp': 80.0, 'glucose': 121.0}`
