import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from src.models.feature_engineering import add_pulse_pressure, add_bmi_age_interaction

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_CHECKUP_FEATURES = ['age', 'sex', 'bmi', 'systolic_bp', 'diastolic_bp', 'glucose', 'cholesterol', 'smoking',
                         'waist_circumference', 'resting_pulse', 'uric_acid']
ENGINEERED_FEATURES = ['pulse_pressure', 'bmi_age_interaction']

# Monotonic constraints (checkup-safe, 13 features):
# age↑ sex(any) bmi↑ sysBP↑ diaBP(any) glucose↑ chol(any) smoking(any)
# waist↑ pulse↑ uric_acid↑ pulse_pressure↑ bmi_age↑
SAFE_MONOTONIC = (1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_pulse_pressure(df)
    df = add_bmi_age_interaction(df)
    return df

def train_and_evaluate():
    # Load processed data. Prefer the NHANES-derived cohort (5k+ adults, both
    # sexes, CKD labelled by CKD-EPI 2021 eGFR + KDIGO albuminuria); fall back to
    # the legacy 400-row UCI set if it has not been built yet.
    data_path = os.path.join(DATA_PROCESSED, "ckd_nhanes_clean.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_PROCESSED, "ckd_clean.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned CKD data not found at {data_path}")
    df = pd.read_csv(data_path)
    df = _engineer(df)

    # Define features. The checkup-safe set is non-invasive routine inputs only.
    # 'serum_creatinine' is included in the baseline for an illustrative
    # "with-lab" comparison, but it is LEAKY (eGFR — hence the label — is derived
    # from it), so it is deliberately excluded from the shipped checkup-safe model.
    checkup_safe_features = RAW_CHECKUP_FEATURES + ENGINEERED_FEATURES
    baseline_features = checkup_safe_features + (['serum_creatinine'] if 'serum_creatinine' in df.columns else [])
    target_col = 'classification'
    
    X_base = df[baseline_features]
    X_safe = df[checkup_safe_features]
    y = df[target_col]
    
    # Calculate training medians from the dataset
    medians = df[checkup_safe_features].median().to_dict()
    
    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Dictionary to store metrics
    metrics = {
        'baseline': {'accuracy': [], 'auc': [], 'f1': [], 'confusion': []},
        'checkup_safe': {'accuracy': [], 'auc': [], 'f1': [], 'confusion': []}
    }
    
    # XGBoost hyperparameters
    base_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss',
        'monotone_constraints': SAFE_MONOTONIC + ((1,) if 'serum_creatinine' in baseline_features else ()),
    }
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss',
        'monotone_constraints': SAFE_MONOTONIC,
    }

    # 5-fold CV for Baseline (Full-feature) Model
    print("Training Baseline Model (5-fold CV)...")
    for train_idx, val_idx in kf.split(X_base):
        X_train, X_val = X_base.iloc[train_idx], X_base.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE to training split only
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(**base_params)
        model.fit(X_train_res, y_train_res)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        metrics['baseline']['accuracy'].append(accuracy_score(y_val, preds))
        metrics['baseline']['auc'].append(roc_auc_score(y_val, probs))
        metrics['baseline']['f1'].append(f1_score(y_val, preds))
        metrics['baseline']['confusion'].append(confusion_matrix(y_val, preds))
        
    # 5-fold CV for Checkup-safe Model
    print("Training Checkup-safe Model (5-fold CV)...")
    for train_idx, val_idx in kf.split(X_safe):
        X_train, X_val = X_safe.iloc[train_idx], X_safe.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply SMOTE to training split only
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        model = XGBClassifier(**xgb_params)
        model.fit(X_train_res, y_train_res)
        
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        metrics['checkup_safe']['accuracy'].append(accuracy_score(y_val, preds))
        metrics['checkup_safe']['auc'].append(roc_auc_score(y_val, probs))
        metrics['checkup_safe']['f1'].append(f1_score(y_val, preds))
        metrics['checkup_safe']['confusion'].append(confusion_matrix(y_val, preds))

    # Calculate average metrics
    base_acc = np.mean(metrics['baseline']['accuracy'])
    base_auc = np.mean(metrics['baseline']['auc'])
    base_f1 = np.mean(metrics['baseline']['f1'])
    base_conf = np.sum(metrics['baseline']['confusion'], axis=0)
    
    safe_acc = np.mean(metrics['checkup_safe']['accuracy'])
    safe_auc = np.mean(metrics['checkup_safe']['auc'])
    safe_f1 = np.mean(metrics['checkup_safe']['f1'])
    safe_conf = np.sum(metrics['checkup_safe']['confusion'], axis=0)
    
    print("\n=== CKD Baseline Model Performance ===")
    print(f"Accuracy: {base_acc:.4f}")
    print(f"ROC AUC:  {base_auc:.4f}")
    print(f"F1-Score: {base_f1:.4f}")
    print("Confusion Matrix:\n", base_conf)
    
    print("\n=== CKD Checkup-Safe Model Performance ===")
    print(f"Accuracy: {safe_acc:.4f}")
    print(f"ROC AUC:  {safe_auc:.4f}")
    print(f"F1-Score: {safe_f1:.4f}")
    print("Confusion Matrix:\n", safe_conf)
    
    # Train final checkup-safe model on the entire dataset with SMOTE
    print("\nTraining final checkup-safe model on entire dataset...")
    smote = SMOTE(random_state=42)
    X_safe_res, y_res = smote.fit_resample(X_safe, y)
    
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X_safe_res, y_res)
    
    # Save the final checkup-safe model along with metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "ckd_model.pkl")
    
    model_data = {
        'model': final_model,
        'features': checkup_safe_features,
        'medians': medians
    }
    joblib.dump(model_data, model_path)
    print(f"Saved checkup-safe model to {model_path}")
    
    return {
        'baseline': (base_acc, base_auc, base_f1, base_conf),
        'checkup_safe': (safe_acc, safe_auc, safe_f1, safe_conf)
    }

# --- HANDOFF PREDICTION INTERFACE ---

def _load_model_data():
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ckd_model.pkl")
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained CKD model file not found at {model_path}. Run training first.")
    return joblib.load(model_path)

def predict_risk(patient_df) -> float:
    """
    Exposes prediction interface for single patient risk scoring.
    Accepts dict, Series, or DataFrame.
    Returns 0-1 probability.
    """
    # Parse input into DataFrame
    if isinstance(patient_df, dict):
        df = pd.DataFrame([patient_df])
    elif isinstance(patient_df, pd.Series):
        df = pd.DataFrame([patient_df])
    else:
        df = patient_df.copy()
        
    # Load model and metadata
    model_data = _load_model_data()
    model = model_data['model']
    features = model_data['features']
    medians = model_data['medians']
    
    # Impute missing raw features using training medians, then derive engineered ones
    for col in RAW_CHECKUP_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer(df)

    # Extract only features model was trained on
    df_feats = df[features]

    # Predict probability
    probs = model.predict_proba(df_feats)
    return float(probs[0, 1])

def predict_risk_batch(patient_df) -> pd.DataFrame:
    """
    Exposes prediction interface for batch patient risk scoring.
    Accepts DataFrame.
    Returns DataFrame with columns [patient_id, ckd_risk].
    """
    df = patient_df.copy()
    
    # Load model and metadata
    model_data = _load_model_data()
    model = model_data['model']
    features = model_data['features']
    medians = model_data['medians']
    
    # Impute missing raw features, then derive engineered ones
    for col in RAW_CHECKUP_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer(df)

    # Extract features
    df_feats = df[features]
    
    # Predict probabilities
    probs = model.predict_proba(df_feats)[:, 1]
    
    # Construct result DataFrame
    res = pd.DataFrame({
        'patient_id': df['patient_id'] if 'patient_id' in df.columns else range(len(df)),
        'ckd_risk': probs
    })
    return res

if __name__ == "__main__":
    train_and_evaluate()
