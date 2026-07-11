# AarogyaDrishti AI - Canonical Feature Schema & Leakage Risk Analysis

This document defines the unified canonical feature schema used across all models and datasets in AarogyaDrishti AI. It documents feature units, expected ranges, raw-to-canonical mappings, and leakage risk categorization.

---

## 1. Canonical Feature Schema

The canonical schema represents the core set of features that can be collected during a routine patient checkup.

| Feature Name | Type | Expected Units / Range | Description |
|---|---|---|---|
| `patient_id` | String | Unique string | Unique identifier for the patient |
| `age` | Float | Years (0.0 to 120.0) | Patient's age in years |
| `sex` | Float | `1.0` = Male, `0.0` = Female | Biological sex |
| `bmi` | Float | kg/m² (10.0 to 100.0) | Body Mass Index |
| `systolic_bp` | Float | mmHg (50.0 to 250.0) | Systolic blood pressure |
| `diastolic_bp` | Float | mmHg (30.0 to 150.0) | Diastolic blood pressure |
| `glucose` | Float | mg/dL (50.0 to 500.0) | Blood glucose level (random/fasting or oral glucose tolerance) |
| `cholesterol` | Float | mg/dL or categorical | Serum cholesterol or cholesterol category flag |
| `smoking` | Float | Binary `1.0` = Smoker, `0.0` = Non-smoker | Active smoking status |
| `alcohol` | Float | Binary `1.0` = Active consumer, `0.0` = Non-consumer | Regular/heavy alcohol consumption |
| `physical_activity`| Float | Binary `1.0` = Active, `0.0` = Inactive | Performs regular physical exercise |
| `family_history` | Float | Float score or binary flag | Genetic risk score or binary family history flag for comorbidity |

---

## 2. Leakage Risk Analysis

To build a reliable early-warning prediction system, we must distinguish between features available at a **routine checkup** versus features only known **post-diagnosis** or during active disease workups (which would cause data leakage).

### Leakage Risk (Post-Diagnosis / Active Workup)
These features are excluded from the checkup-safe inference pipeline:
- **Insulin (Diabetes Dataset)**: Often indicates active insulin therapy or advanced diagnostic tests rather than baseline checkup data.
- **Serum Creatinine (`sc`) & Blood Urea (`bu`) (CKD Dataset)**: Direct biochemical markers of kidney damage collected during an active kidney workup.
- **Hemoglobin (`hemo`) & Packed Cell Volume (`pcv`) (CKD Dataset)**: Lab markers of anemia, which are clinical consequences of CKD (post-diagnosis).
- **Specific Gravity (`sg`), Albumin (`al`), Sugar (`su`), RBC/Pus Cell/Bacteria flags (CKD Dataset)**: Urine lab analysis markers specific to diagnosing active kidney pathology.

### Checkup-Safe (Early-Warning)
These features are safe to use for early-warning risk scoring:
- Demographics: `age`, `sex`
- Vitals: `systolic_bp`, `diastolic_bp`, `bmi`
- Basic blood work: `glucose`, `cholesterol`
- Lifestyle surveys: `smoking`, `alcohol`, `physical_activity`
- Family history: `family_history`

---

## 3. Dataset Raw-to-Canonical Column Mapping

Below is the column mapping for all 8 raw CSV files in `data/raw/` to map them onto the canonical schema.

### 1. `diabetes.csv` (Pima Indians Diabetes)
- `patient_id` $\rightarrow$ Generated index-based string (`pima_{index}`)
- `age` $\rightarrow$ `Age` (years)
- `sex` $\rightarrow$ Set to `0.0` (all female in Pima Indians dataset)
- `bmi` $\rightarrow$ `BMI` (kg/m²)
- `diastolic_bp` $\rightarrow$ `BloodPressure` (mmHg)
- `glucose` $\rightarrow$ `Glucose` (2-hour plasma glucose concentration)
- `family_history` $\rightarrow$ `DiabetesPedigreeFunction` (family history score)
- *Others (systolic_bp, cholesterol, smoking, alcohol, physical_activity)* $\rightarrow$ `NaN`
- *Target* $\rightarrow$ `Outcome` (`1` = Diabetes, `0` = No Diabetes)
- *Excluded (Leakage Risk)* $\rightarrow$ `Insulin`, `SkinThickness`

### 2. `kidney_disease.csv` (UCI Chronic Kidney Disease)
- `patient_id` $\rightarrow$ `id`
- `age` $\rightarrow$ `age` (years)
- `diastolic_bp` $\rightarrow$ `bp` (mmHg)
- `glucose` $\rightarrow$ `bgr` (Blood Glucose Random, mg/dL)
- *Others (sex, bmi, systolic_bp, cholesterol, smoking, alcohol, physical_activity, family_history)* $\rightarrow$ `NaN`
- *Target* $\rightarrow$ `classification` (Cleaned: `1` = CKD/`ckd`/`ckd\t`, `0` = Not CKD/`notckd`)
- *Excluded (Leakage Risk)* $\rightarrow$ `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`

### 3. `heart.csv` (Cleveland/UCI Heart Disease)
- `age` $\rightarrow$ `age` (years)
- `sex` $\rightarrow$ `sex` (`1.0` = Male, `0.0` = Female)
- `systolic_bp` $\rightarrow$ `trestbps` (resting blood pressure, mmHg)
- `cholesterol` $\rightarrow$ `chol` (serum cholesterol, mg/dL)
- *Others* $\rightarrow$ `NaN`
- *Target* $\rightarrow$ `target` (`1` = Disease, `0` = Normal)

### 4. `hypertension.csv` (Cardiovascular Disease Dataset)
- `patient_id` $\rightarrow$ `id`
- `age` $\rightarrow$ `age` / 365.25 (converted from days to years)
- `sex` $\rightarrow$ `gender` (Mapped: `2` $\rightarrow$ `1.0` [Male], `1` $\rightarrow$ `0.0` [Female])
- `bmi` $\rightarrow$ `weight` / (`height` / 100)² (calculated from weight in kg and height in cm)
- `systolic_bp` $\rightarrow$ `ap_hi` (mmHg)
- `diastolic_bp` $\rightarrow$ `ap_lo` (mmHg)
- `cholesterol` $\rightarrow$ `cholesterol` (coded `1`, `2`, `3`)
- `glucose` $\rightarrow$ `gluc` (coded `1`, `2`, `3`)
- `smoking` $\rightarrow$ `smoke` (binary)
- `alcohol` $\rightarrow$ `alco` (binary)
- `physical_activity` $\rightarrow$ `active` (binary)
- *Target* $\rightarrow$ `cardio` (`1` = Cardiovascular disease, `0` = Healthy)

### 5. `heart + hypertension.csv` (Stroke/Comorbidity Dataset)
- `patient_id` $\rightarrow$ `id`
- `age` $\rightarrow$ `age` (years)
- `sex` $\rightarrow$ `gender` (Mapped: `'Male'` $\rightarrow$ `1.0`, `'Female'` $\rightarrow$ `0.0`, `'Other'` $\rightarrow$ `NaN`)
- `bmi` $\rightarrow$ `bmi` (kg/m²)
- `glucose` $\rightarrow$ `avg_glucose_level` (mg/dL)
- `smoking` $\rightarrow$ `smoking_status` (Mapped: `'smokes'`/`'formerly smoked'` $\rightarrow$ `1.0`, `'never smoked'` $\rightarrow$ `0.0`, `'Unknown'` $\rightarrow$ `NaN`)
- *Others* $\rightarrow$ `NaN`
- *Target/Co-labels* $\rightarrow$ `hypertension` (binary), `heart_disease` (binary), `stroke` (binary)

### 6. `dia + heart + kidney.csv` (BRFSS 2015)
- `age` $\rightarrow$ `Age` (13-level age category scale)
- `sex` $\rightarrow$ `Sex` (`1.0` = Male, `0.0` = Female)
- `bmi` $\rightarrow$ `BMI`
- `smoking` $\rightarrow$ `Smoker` (binary)
- `alcohol` $\rightarrow$ `HvyAlcoholConsump` (binary)
- `physical_activity` $\rightarrow$ `PhysActivity` (binary)
- *Others* $\rightarrow$ `NaN`
- *Target/Co-labels* $\rightarrow$ `Diabetes` (0/1/2), `HighBP` (binary), `HeartDiseaseorAttack` (binary)

### 7. `diabetes + hypertension 1.csv` (BRFSS Subset 1)
- Similar mapping to BRFSS 2015 above.

### 8. `diabetes + HT 2.csv` (Heart/Hypertension Subset 2)
- Same schema and mapping as `heart.csv`.
