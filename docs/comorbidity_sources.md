# Comorbidity Ground Truth Verification

This document verifies that the comorbidity-labeled datasets in the repository contain real co-occurring target labels for multiple diseases at the individual patient level.

---

## 1. `heart + hypertension.csv` (Stroke/Comorbidity Dataset)

* **Total Rows**: 5,110 patients
* **Target Columns**: `hypertension` (binary 0/1) and `heart_disease` (binary 0/1)
* **Co-occurrence Analysis**:
  * Patients with **both** Hypertension & Heart Disease: **64**
  * Patients with Hypertension **only**: **434**
  * Patients with Heart Disease **only**: **212**
  * Patients with **neither**: **4,400**

### Conclusion
This dataset is a valid source of real co-occurring labels for Hypertension and Heart Disease. The prevalence of heart disease among hypertensive patients is **12.85%** ($64 / (434+64)$), compared to only **4.60%** ($212 / (4400+212)$) among non-hypertensive patients. This represents a **2.79× risk increase**, which is clinically consistent.

---

## 2. `dia + heart + kidney.csv` (BRFSS 2015)

* **Total Rows**: 253,680 patients
* **Target Columns**: `Diabetes` (0 = No, 1 = Prediabetes, 2 = Diabetes), `HighBP` (binary 0/1), and `HeartDiseaseorAttack` (binary 0/1)
* **CKD Specific Column Verification**: 
  * Checked for kidney, CKD, or renal related columns in the dataset.
  * **Result**: **No CKD-specific columns exist**, confirming the known quirk. This dataset will be used solely for Diabetes, Hypertension, and Heart Disease co-labels.
* **Co-occurrence Analysis (Hypertension vs. Diabetes)**:
  * Patients with **HighBP (1.0) & Diabetes (2.0)**: **26,604**
  * Patients with **HighBP (1.0) only**: **79,312** (No Diabetes) + **2,913** (Prediabetes) = **82,225**
  * Patients with **Diabetes (2.0) only**: **8,742** (No HighBP)

### Conclusion
This dataset is a massive, high-quality source of co-occurring labels for Diabetes, Hypertension, and Heart Disease. The high co-occurrence rate of HighBP and Diabetes (26,604 patients) represents a major overlap of metabolic syndrome components, which is essential for training the chained risk models.
