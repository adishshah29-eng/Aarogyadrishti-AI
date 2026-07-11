import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

def train_and_evaluate():
    # Load processed data
    data_path = r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\data\processed\diabetes_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned diabetes data not found at {data_path}")
    df = pd.read_csv(data_path)
    
    # Define features
    baseline_features = ['Pregnancies', 'glucose', 'diastolic_bp', 'SkinThickness', 'Insulin', 'bmi', 'family_history', 'age']
    checkup_safe_features = ['age', 'sex', 'bmi', 'diastolic_bp', 'glucose', 'family_history']
    target_col = 'Outcome'
    
    X_base = df[baseline_features]
    X_safe = df[checkup_safe_features]
    y = df[target_col]
    
    # Calculate training medians from the dataset (excluding target and ID)
    medians = df[checkup_safe_features].median().to_dict()
    
    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Dictionary to store metrics
    metrics = {
        'baseline': {'accuracy': [], 'auc': [], 'f1': [], 'confusion': []},
        'checkup_safe': {'accuracy': [], 'auc': [], 'f1': [], 'confusion': []}
    }
    
    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # 5-fold CV for Baseline (Full-feature) Model
    print("Training Baseline Model (5-fold CV)...")
    for train_idx, val_idx in kf.split(X_base):
        X_train, X_val = X_base.iloc[train_idx], X_base.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply SMOTE to training split only
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        model = XGBClassifier(**xgb_params)
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
    
    print("\n=== Diabetes Baseline Model Performance ===")
    print(f"Accuracy: {base_acc:.4f}")
    print(f"ROC AUC:  {base_auc:.4f}")
    print(f"F1-Score: {base_f1:.4f}")
    print("Confusion Matrix:\n", base_conf)
    
    print("\n=== Diabetes Checkup-Safe Model Performance ===")
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
    
    # Train final baseline model on entire dataset for comparison/logging
    smote_b = SMOTE(random_state=42)
    X_base_res, y_res_b = smote_b.fit_resample(X_base, y)
    final_base_model = XGBClassifier(**xgb_params)
    final_base_model.fit(X_base_res, y_res_b)
    
    # Save the final checkup-safe model along with metadata
    models_dir = r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "diabetes_model.pkl")
    
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
    model_path = r"c:\Users\Sayli\OneDrive\Desktop\Aarogyadrishti-AI\models\diabetes_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained diabetes model file not found at {model_path}. Run training first.")
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
    
    # Impute missing features using training medians
    for col in features:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
            
    # Extract only features model was trained on
    df_feats = df[features]
    
    # Predict probability
    probs = model.predict_proba(df_feats)
    return float(probs[0, 1])

def predict_risk_batch(patient_df) -> pd.DataFrame:
    """
    Exposes prediction interface for batch patient risk scoring.
    Accepts DataFrame.
    Returns DataFrame with columns [patient_id, diabetes_risk].
    """
    df = patient_df.copy()
    
    # Load model and metadata
    model_data = _load_model_data()
    model = model_data['model']
    features = model_data['features']
    medians = model_data['medians']
    
    # Impute missing features
    for col in features:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
            
    # Extract features
    df_feats = df[features]
    
    # Predict probabilities
    probs = model.predict_proba(df_feats)[:, 1]
    
    # Construct result DataFrame
    res = pd.DataFrame({
        'patient_id': df['patient_id'] if 'patient_id' in df.columns else range(len(df)),
        'diabetes_risk': probs
    })
    return res

if __name__ == "__main__":
    train_and_evaluate()
