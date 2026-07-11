import os
import pandas as pd
import numpy as np

def clean_diabetes(path: str) -> pd.DataFrame:
    """
    Clean the Diabetes (Pima) dataset.
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
    Clean the Cleveland Heart Disease dataset.
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
    
    df_clean['cholesterol'] = df['cholesterol'].astype(float)
    df_clean['glucose'] = df['gluc'].astype(float)
    
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
    # Create processed directory if it doesn't exist
    processed_dir = r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\data\processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Cleaning diabetes dataset...")
    df_dia = clean_diabetes(r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\data\raw\diabetes.csv")
    df_dia.to_csv(os.path.join(processed_dir, "diabetes_clean.csv"), index=False)
    print(f"Saved cleaned diabetes dataset to processed/diabetes_clean.csv (Shape: {df_dia.shape})")
    
    print("\nCleaning CKD dataset...")
    df_ckd = clean_ckd(r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\data\raw\kidney_disease.csv")
    df_ckd.to_csv(os.path.join(processed_dir, "ckd_clean.csv"), index=False)
    print(f"Saved cleaned CKD dataset to processed/ckd_clean.csv (Shape: {df_ckd.shape})")

    print("\nCleaning Heart disease dataset...")
    df_heart = clean_heart(r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\data\raw\heart.csv")
    df_heart.to_csv(os.path.join(processed_dir, "heart_clean.csv"), index=False)
    print(f"Saved cleaned heart dataset to processed/heart_clean.csv (Shape: {df_heart.shape})")

    print("\nCleaning Hypertension dataset...")
    df_ht = clean_hypertension(r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\data\raw\hypertension.csv")
    df_ht.to_csv(os.path.join(processed_dir, "hypertension_clean.csv"), index=False)
    print(f"Saved cleaned hypertension dataset to processed/hypertension_clean.csv (Shape: {df_ht.shape})")

