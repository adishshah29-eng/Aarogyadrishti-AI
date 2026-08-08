import os
import pandas as pd
import numpy as np

def clean_diabetes(path: str) -> pd.DataFrame:
    """
    Clean the 100k mixed-sex Diabetes Prediction dataset (Kaggle: iammustafatz,
    CC0). Columns: gender, age, hypertension, heart_disease, smoking_history,
    bmi, HbA1c_level, blood_glucose_level, diabetes.

    This replaces the female-only Pima dataset as the diabetes training source
    so the model is validated across both sexes. Steps:
    - Drop exact duplicate rows and restrict to adults (age >= 18) to match a
      screening context.
    - Map gender -> sex (Male=1.0, Female=0.0); drop the handful of 'Other' rows.
    - Map columns onto the canonical schema; clip implausible BMI.
    - Binarize smoking_history into a current/active-smoking flag.
    - Keep HbA1c_level, hypertension, heart_disease as extra columns for the
      full-feature baseline only (NOT used by the shipped checkup-safe model:
      HbA1c is a diagnostic marker, and the two comorbidity labels would create
      circularity in the upstream->downstream chain).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw diabetes file not found at {path}")

    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df[df['age'] >= 18].copy()

    # Map gender -> sex; 'Other' (a few rows) has no clear encoding, so drop it.
    df['sex_mapped'] = df['gender'].map({'Male': 1.0, 'Female': 0.0})
    df = df[df['sex_mapped'].notna()].copy()
    # Reset the (now gappy) index so Series assignments below align positionally
    # into the fresh df_clean rather than by mismatched index labels.
    df = df.reset_index(drop=True)

    # Binarize smoking history into an active/ever-smoking risk flag.
    # 'No Info' is unknown -> treated as non-smoker (the majority class).
    smoke_map = {
        'never': 0.0, 'No Info': 0.0, 'not current': 0.0,
        'former': 1.0, 'current': 1.0, 'ever': 1.0,
    }
    smoking = df['smoking_history'].map(smoke_map).fillna(0.0)

    df_clean = pd.DataFrame()
    df_clean['patient_id'] = [f"dpd_{i}" for i in range(len(df))]
    df_clean['age'] = df['age'].astype(float)
    df_clean['sex'] = df['sex_mapped'].astype(float)
    df_clean['bmi'] = df['bmi'].astype(float).clip(13.0, 60.0)
    df_clean['systolic_bp'] = np.nan
    df_clean['diastolic_bp'] = np.nan
    df_clean['glucose'] = df['blood_glucose_level'].astype(float)
    df_clean['cholesterol'] = np.nan
    df_clean['smoking'] = smoking.astype(float)
    df_clean['alcohol'] = np.nan
    df_clean['physical_activity'] = np.nan
    df_clean['family_history'] = np.nan  # not available in this dataset

    # Extra features retained for the full-feature baseline model only.
    df_clean['HbA1c_level'] = df['HbA1c_level'].astype(float)
    df_clean['hypertension'] = df['hypertension'].astype(float)
    df_clean['heart_disease'] = df['heart_disease'].astype(float)

    # Target (kept as 'Outcome' for interface compatibility)
    df_clean['Outcome'] = df['diabetes'].astype(int)

    return df_clean.reset_index(drop=True)


def clean_diabetes_pima(path: str) -> pd.DataFrame:
    """
    Clean the legacy Diabetes (Pima Indians) dataset. Retained for provenance;
    the shipped diabetes model now trains on the 100k mixed-sex dataset via
    clean_diabetes(). The Pima cohort is female-only.
    - Treat invalid 0s in Glucose, BloodPressure, SkinThickness, Insulin, BMI as NaN.
    - Map columns to the canonical schema.
    - Perform median imputation.
    - Return cleaned DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw diabetes file not found at {path}")
        
    df = pd.read_csv(path)
    
    # 1. Handle biological invalid zeros (replace with NaN)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            
    # 2. Impute missing values with median
    # We compute medians on the dataset itself for pipeline cleanliness,
    # but store training medians for final prediction-time imputation.
    for col in zero_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # 3. Create canonical schema columns
    df_clean = pd.DataFrame()
    
    # Generate patient_id
    df_clean['patient_id'] = [f"pima_{i}" for i in range(len(df))]
    
    # Direct mappings
    df_clean['age'] = df['Age'].astype(float)
    df_clean['sex'] = 0.0  # Pima Indians dataset contains only females
    df_clean['bmi'] = df['BMI'].astype(float)
    df_clean['systolic_bp'] = np.nan
    df_clean['diastolic_bp'] = df['BloodPressure'].astype(float)
    df_clean['glucose'] = df['Glucose'].astype(float)
    df_clean['cholesterol'] = np.nan
    df_clean['smoking'] = np.nan
    df_clean['alcohol'] = np.nan
    df_clean['physical_activity'] = np.nan
    df_clean['family_history'] = df['DiabetesPedigreeFunction'].astype(float)
    
    # Keep original raw features for full-feature baseline training
    df_clean['Pregnancies'] = df['Pregnancies'].astype(float)
    df_clean['SkinThickness'] = df['SkinThickness'].astype(float)
    df_clean['Insulin'] = df['Insulin'].astype(float)
    
    # Add target
    df_clean['Outcome'] = df['Outcome'].astype(int)
    
    return df_clean

def clean_ckd(path: str) -> pd.DataFrame:
    """
    Clean the UCI Chronic Kidney Disease dataset.
    - Handle string typos and whitespaces.
    - Convert pcv, wc, rc to float (turning '\t?' into NaN).
    - Map categorical columns to binary numeric (0/1).
    - Map columns to the canonical schema.
    - Perform median (for numeric) and mode (for categorical) imputation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw CKD file not found at {path}")
        
    df = pd.read_csv(path)
    
    # Clean up column names and string values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.replace(r'\t', '', regex=True)
            
    # 1. Clean target: classification
    # Classification values: 'ckd', 'notckd', 'nan' (if any)
    # Map 'ckd' -> 1, 'notckd' -> 0
    df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})
    # Drop rows if classification is missing or invalid
    df = df[df['classification'].isin([0, 1])].copy()
    df['classification'] = df['classification'].astype(int)
    
    # 2. Convert numeric columns stored as objects due to quirks (pcv, wc, rc)
    for col in ['pcv', 'wc', 'rc']:
        # Replace '?' or empty string with NaN
        df[col] = df[col].replace({'?': np.nan, '': np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 3. Clean and map categorical/binary columns
    # yes/no columns
    yes_no_cols = ['htn', 'dm', 'cad', 'pe', 'ane']
    for col in yes_no_cols:
        df[col] = df[col].replace({'yes': 1.0, 'no': 0.0, '?': np.nan, 'nan': np.nan})
        
    # appet: good/poor
    df['appet'] = df['appet'].replace({'good': 1.0, 'poor': 0.0, '?': np.nan, 'nan': np.nan})
    
    # rbc, pc: normal/abnormal
    for col in ['rbc', 'pc']:
        df[col] = df[col].replace({'normal': 1.0, 'abnormal': 0.0, '?': np.nan, 'nan': np.nan})
        
    # pcc, ba: present/notpresent
    for col in ['pcc', 'ba']:
        df[col] = df[col].replace({'present': 1.0, 'notpresent': 0.0, '?': np.nan, 'nan': np.nan})
        
    # 4. Perform Imputation
    # Numeric columns
    num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Categorical columns
    cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for col in cat_cols:
        if col in df.columns:
            # Mode returns a Series, take first element or default to 0.0 if empty
            modes = df[col].mode()
            mode_val = modes.iloc[0] if not modes.empty else 0.0
            df[col] = df[col].fillna(mode_val)
            
    # 5. Create canonical schema columns
    df_clean = pd.DataFrame()
    
    # patient_id (use 'id' or generated id if not present)
    df_clean['patient_id'] = df['id'].astype(str)
    
    # Direct mappings
    df_clean['age'] = df['age'].astype(float)
    df_clean['sex'] = np.nan
    df_clean['bmi'] = np.nan
    df_clean['systolic_bp'] = np.nan
    df_clean['diastolic_bp'] = df['bp'].astype(float)
    df_clean['glucose'] = df['bgr'].astype(float)
    df_clean['cholesterol'] = np.nan
    df_clean['smoking'] = np.nan
    df_clean['alcohol'] = np.nan
    df_clean['physical_activity'] = np.nan
    df_clean['family_history'] = np.nan
    
    # Keep original raw features for full-feature baseline training
    raw_extra_features = [
        'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bu', 'sc', 'sod', 'pot', 
        'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]
    for feat in raw_extra_features:
        df_clean[feat] = df[feat].astype(float)
        
    # Target
    df_clean['classification'] = df['classification'].astype(int)
    
    return df_clean

def clean_heart(path: str) -> pd.DataFrame:
    """
    Clean the Framingham Heart Study dataset (10-year CHD outcome).

    This replaces the legacy 1,025-row Kaggle heart dataset, whose checkup
    features (age, blood pressure, cholesterol) were **inversely** correlated
    with the label — older / higher-BP / higher-cholesterol patients were
    coded as *lower* risk, producing clinically backwards predictions. In
    Framingham these relationships are correct (age/sysBP/totChol/glucose all
    correlate positively with 10-year CHD), and the checkup-friendly columns
    (age, sex, BMI, systolic/diastolic BP, glucose, total cholesterol, smoking)
    map directly onto the canonical schema.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw heart file not found at {path}")

    df = pd.read_csv(path)

    df_clean = pd.DataFrame()
    df_clean['patient_id'] = [f"fram_{i}" for i in range(len(df))]
    df_clean['age'] = df['age'].astype(float)
    df_clean['sex'] = df['male'].astype(float)          # 1 = male, 0 = female
    df_clean['bmi'] = df['BMI'].astype(float)
    df_clean['systolic_bp'] = df['sysBP'].astype(float)
    df_clean['diastolic_bp'] = df['diaBP'].astype(float)
    df_clean['glucose'] = df['glucose'].astype(float)
    df_clean['cholesterol'] = df['totChol'].astype(float)
    df_clean['smoking'] = df['currentSmoker'].astype(float)
    df_clean['alcohol'] = np.nan
    df_clean['physical_activity'] = np.nan
    df_clean['family_history'] = np.nan

    # Extra Framingham columns kept for the full-feature baseline comparison.
    for col in ['heartRate', 'cigsPerDay', 'BPMeds', 'prevalentHyp', 'diabetes', 'prevalentStroke']:
        df_clean[col] = df[col].astype(float)

    df_clean['target'] = df['TenYearCHD'].astype(int)

    # Median-impute any missing feature values so SMOTE/XGBoost have clean input.
    for col in df_clean.columns:
        if col in ('patient_id', 'target'):
            continue
        if df_clean[col].isnull().any():
            med = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med if pd.notna(med) else 0.0)

    return df_clean


def clean_heart_cleveland(path: str) -> pd.DataFrame:
    """
    Legacy cleaner for the 1,025-row Kaggle/Cleveland heart dataset. Retained
    for provenance; superseded by clean_heart() (Framingham) because this
    dataset's checkup features are inversely correlated with the label.
    - Map columns to the canonical schema.
    - Perform median imputation for missing values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw heart file not found at {path}")
        
    df = pd.read_csv(path)
    
    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Create canonical schema columns
    df_clean = pd.DataFrame()
    
    # patient_id
    df_clean['patient_id'] = [f"heart_{i}" for i in range(len(df))]
    
    # Direct mappings
    df_clean['age'] = df['age'].astype(float)
    df_clean['sex'] = df['sex'].astype(float)
    df_clean['bmi'] = np.nan
    df_clean['systolic_bp'] = df['trestbps'].astype(float)
    df_clean['diastolic_bp'] = np.nan
    df_clean['glucose'] = np.nan  # fbs is binary, not continuous
    df_clean['cholesterol'] = df['chol'].astype(float)
    df_clean['smoking'] = np.nan
    df_clean['alcohol'] = np.nan
    df_clean['physical_activity'] = np.nan
    df_clean['family_history'] = np.nan
    
    # Keep extra raw features for baseline model
    raw_extra = ['cp', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    for col in raw_extra:
        df_clean[col] = df[col].astype(float)
        
    # Target
    df_clean['target'] = df['target'].astype(int)
    
    return df_clean

def clean_hypertension(path: str) -> pd.DataFrame:
    """
    Clean the Cardiovascular Disease (hypertension proxy) dataset.
    - Handle semicolon separator.
    - Convert age from days to years.
    - Map gender (2=Male, 1=Female) -> sex (1.0=Male, 0.0=Female).
    - Calculate BMI from height and weight.
    - Clean and clip blood pressure outliers.
    - Map to canonical schema.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw hypertension file not found at {path}")
        
    df = pd.read_csv(path, sep=';')
    
    # Convert age from days to years
    df['age_years'] = df['age'] / 365.25
    
    # Map gender (2 = Male, 1 = Female) -> sex (1.0 = Male, 0.0 = Female)
    df['sex_mapped'] = df['gender'].replace({2: 1.0, 1: 0.0})
    
    # Calculate BMI
    df['bmi_calc'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['bmi_calc'] = df['bmi_calc'].clip(10.0, 100.0)
    
    # Clean blood pressure outliers: ap_hi (systolic) and ap_lo (diastolic)
    df.loc[(df['ap_hi'] < 60) | (df['ap_hi'] > 250), 'ap_hi'] = np.nan
    df.loc[(df['ap_lo'] < 40) | (df['ap_lo'] > 150), 'ap_lo'] = np.nan
    
    # If systolic < diastolic, set to NaN
    invalid_bp = df['ap_hi'] < df['ap_lo']
    df.loc[invalid_bp, 'ap_hi'] = np.nan
    df.loc[invalid_bp, 'ap_lo'] = np.nan
    
    # Perform median imputation
    df['ap_hi'] = df['ap_hi'].fillna(df['ap_hi'].median())
    df['ap_lo'] = df['ap_lo'].fillna(df['ap_lo'].median())
    
    # Create canonical schema columns
    df_clean = pd.DataFrame()
    
    # patient_id
    df_clean['patient_id'] = df['id'].astype(str)
    
    # Direct mappings
    df_clean['age'] = df['age_years'].astype(float)
    df_clean['sex'] = df['sex_mapped'].astype(float)
    df_clean['bmi'] = df['bmi_calc'].astype(float)
    df_clean['systolic_bp'] = df['ap_hi'].astype(float)
    df_clean['diastolic_bp'] = df['ap_lo'].astype(float)
    
    # The cardio dataset codes cholesterol/glucose as ordinal categories
    # (1 = normal, 2 = above normal, 3 = well above normal). Every other dataset
    # — and the dashboard — uses clinical mg/dL. Map the categories to
    # representative mg/dL midpoints so the canonical schema is unit-consistent
    # across all models (see src/schema.py). Without this the hypertension model
    # would be trained on 1-3 while receiving 100-400 at inference, saturating
    # the trees and making cholesterol inert.
    chol_to_mgdl = {1.0: 180.0, 2.0: 220.0, 3.0: 270.0}   # normal / borderline-high / high
    gluc_to_mgdl = {1.0: 90.0, 2.0: 115.0, 3.0: 160.0}    # normal / impaired / diabetic-range
    df_clean['cholesterol'] = df['cholesterol'].astype(float).map(chol_to_mgdl)
    df_clean['glucose'] = df['gluc'].astype(float).map(gluc_to_mgdl)

    df_clean['smoking'] = df['smoke'].astype(float)
    df_clean['alcohol'] = df['alco'].astype(float)
    df_clean['physical_activity'] = df['active'].astype(float)
    df_clean['family_history'] = np.nan
    
    # Keep extra raw features
    df_clean['height'] = df['height'].astype(float)
    df_clean['weight'] = df['weight'].astype(float)
    
    # Target
    df_clean['cardio'] = df['cardio'].astype(int)
    
    return df_clean

if __name__ == "__main__":
    # Resolve paths relative to the repository root so the pipeline runs
    # anywhere, not just on the original author's machine.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("Cleaning diabetes dataset (100k mixed-sex)...")
    df_dia = clean_diabetes(os.path.join(raw_dir, "diabetes_100k.csv"))
    df_dia.to_csv(os.path.join(processed_dir, "diabetes_clean.csv"), index=False)
    print(f"Saved cleaned diabetes dataset to processed/diabetes_clean.csv (Shape: {df_dia.shape})")

    print("\nCleaning CKD dataset...")
    df_ckd = clean_ckd(os.path.join(raw_dir, "kidney_disease.csv"))
    df_ckd.to_csv(os.path.join(processed_dir, "ckd_clean.csv"), index=False)
    print(f"Saved cleaned CKD dataset to processed/ckd_clean.csv (Shape: {df_ckd.shape})")

    print("\nCleaning Heart disease dataset (Framingham)...")
    df_heart = clean_heart(os.path.join(raw_dir, "framingham.csv"))
    df_heart.to_csv(os.path.join(processed_dir, "heart_clean.csv"), index=False)
    print(f"Saved cleaned heart dataset to processed/heart_clean.csv (Shape: {df_heart.shape})")

    print("\nCleaning Hypertension dataset...")
    df_ht = clean_hypertension(os.path.join(raw_dir, "hypertension.csv"))
    df_ht.to_csv(os.path.join(processed_dir, "hypertension_clean.csv"), index=False)
    print(f"Saved cleaned hypertension dataset to processed/hypertension_clean.csv (Shape: {df_ht.shape})")

