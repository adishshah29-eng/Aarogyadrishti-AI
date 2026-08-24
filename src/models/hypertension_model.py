"""
Hypertension risk model (downstream / chained).

Training cohort: NHANES 2021-2023 (8,139 adults, both sexes, ~36.3%
positive), built by ``scripts/build_hypertension_nhanes.py``. This replaces
the cardio/Kaggle dataset (70k rows), whose target ('cardio') was a broad
self-reported cardiovascular-disease flag rather than hypertension
specifically, and whose cholesterol/glucose were ordinal categories (1-3)
rather than continuous lab values.

Like the Heart Disease model, this consumes the upstream Diabetes and CKD
risk scores as extra features (``diabetes_risk``, ``ckd_risk``).

NOTE: the cardio dataset's ``alcohol`` and ``physical_activity`` fields are
NOT carried over — no NHANES alcohol-use or physical-activity module is
available in this environment (see scripts/build_hypertension_nhanes.py), and
no other shipped model consumes them, so keeping them would mean asking users
a question whose answer is never actually used.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.upstream import add_upstream_risks, UPSTREAM_FEATURES
from src.models.feature_engineering import add_pulse_pressure, add_mean_arterial_pressure, add_comorbidity_risk_interaction
from src.models.tuned_params import load_tuned

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# XGBoost hyperparameters
XGB_PARAMS = {
    **load_tuned('hypertension', {
        'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8,
    }),
    'random_state': 42,
    'eval_metric': 'logloss',
}

RAW_ISOLATED_FEATURES = ['age', 'sex', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose', 'smoking',
                          'waist_circumference', 'resting_pulse', 'uric_acid', 'bun', 'triglycerides', 'cigs_per_day']
ENGINEERED_FEATURES = ['pulse_pressure', 'mean_arterial_pressure']
# Computed only once diabetes_risk/ckd_risk are present (see _engineer_chained).
# SHAP interaction analysis (scripts/analyze_shap_interactions.py) found
# diabetes_risk x ckd_risk to be this model's single strongest feature-pair
# interaction — comorbid risk compounds rather than adds, echoing the CRI
# formula's own interaction terms (src/chaining/cri.py).
CHAINED_ENGINEERED_FEATURES = ['comorbidity_risk_interaction']


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_pulse_pressure(df)
    df = add_mean_arterial_pressure(df)
    return df


def _engineer_chained(df: pd.DataFrame) -> pd.DataFrame:
    return add_comorbidity_risk_interaction(df)


# Monotonic constraints for the shipped chained feature set (19 features):
# age↑ sex(any) bmi↑ sysBP↑ diaBP↑ chol↑ glucose↑ smoking(any)
# waist↑ pulse↑ uric_acid↑ bun↑ triglycerides↑ cigs↑ pulse_pressure↑ MAP↑ dia_risk↑ ckd_risk↑ comorbidity_interaction↑
CHAINED_MONOTONIC = (1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)


def _run_cv(X, y, xgb_params=XGB_PARAMS, n_splits=5):
    """5-fold CV with SMOTE applied on the training split only. Returns mean
    accuracy/AUC/F1 and the summed confusion matrix."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs, f1s, confs = [], [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(**xgb_params)
        model.fit(X_train_res, y_train_res)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        accs.append(accuracy_score(y_val, preds))
        aucs.append(roc_auc_score(y_val, probs))
        f1s.append(f1_score(y_val, preds))
        confs.append(confusion_matrix(y_val, preds))
    return {
        'accuracy': float(np.mean(accs)),
        'auc': float(np.mean(aucs)),
        'f1': float(np.mean(f1s)),
        'confusion': np.sum(confs, axis=0),
    }


def train_and_evaluate():
    # Load processed data. Prefer the NHANES-derived cohort (both sexes,
    # doctor-diagnosed hypertension label, continuous labs); fall back to the
    # legacy cardio/Kaggle set if it has not been built yet.
    data_path = os.path.join(DATA_PROCESSED, "hypertension_nhanes_clean.csv")
    legacy = not os.path.exists(data_path)
    if legacy:
        data_path = os.path.join(DATA_PROCESSED, "hypertension_clean.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned hypertension data not found at {data_path}")
    df = pd.read_csv(data_path)
    df = _engineer(df)

    # Append genuine upstream risk scores so the model can chain on them.
    df = add_upstream_risks(df)
    df = _engineer_chained(df)

    # Define feature sets. Unlike Heart Disease, this dataset has no separate
    # "full lab panel" beyond the checkup-safe set, so there is no meaningful
    # baseline variant to compare against here.
    isolated_features = RAW_ISOLATED_FEATURES + ENGINEERED_FEATURES
    chained_features = isolated_features + UPSTREAM_FEATURES + CHAINED_ENGINEERED_FEATURES   # SHIPPED feature set
    target_col = 'Outcome' if not legacy else 'cardio'

    y = df[target_col]

    print("Hypertension — Isolated checkup-safe 5-fold CV...")
    m_iso = _run_cv(df[isolated_features], y)
    print("Hypertension — Chained checkup-safe 5-fold CV...")
    chained_params = {**XGB_PARAMS, 'monotone_constraints': CHAINED_MONOTONIC}
    m_chain = _run_cv(df[chained_features], y, xgb_params=chained_params)

    for label, m in [("Isolated checkup-safe", m_iso),
                     ("Chained checkup-safe (SHIPS)", m_chain)]:
        print(f"\n=== Hypertension — {label} ===")
        print(f"Accuracy: {m['accuracy']:.4f}")
        print(f"ROC AUC:  {m['auc']:.4f}")
        print(f"F1-Score: {m['f1']:.4f}")
        print("Confusion Matrix:\n", m['confusion'])

    d_acc = m_chain['accuracy'] - m_iso['accuracy']
    d_auc = m_chain['auc'] - m_iso['auc']
    print(f"\nChaining delta (chained - isolated): Accuracy {d_acc:+.4f}, ROC AUC {d_auc:+.4f}")

    # Train final SHIPPED model on all data using the chained feature set.
    medians = df[chained_features].median().to_dict()
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(df[chained_features], y)
    final_model = XGBClassifier(**{**XGB_PARAMS, 'monotone_constraints': CHAINED_MONOTONIC})
    final_model.fit(X_res, y_res)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "hypertension_model.pkl")
    joblib.dump({
        'model': final_model,
        'features': chained_features,
        'medians': medians,
    }, model_path)
    print(f"Saved chained checkup-safe Hypertension model to {model_path}")

    return {'isolated': m_iso, 'chained': m_chain}

# --- HANDOFF PREDICTION INTERFACE ---

def _load_model_data():
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hypertension_model.pkl")
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained hypertension model file not found at {model_path}. Run training first.")
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
    for col in RAW_ISOLATED_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer(df)

    # Resolve upstream chained risks (fall back to medians if not chained),
    # then derive the interaction terms that depend on them.
    for col in UPSTREAM_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer_chained(df)

    # Any remaining feature falls back to its training median too.
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
    Returns DataFrame with columns [patient_id, hypertension_risk].
    """
    df = patient_df.copy()
    
    # Load model and metadata
    model_data = _load_model_data()
    model = model_data['model']
    features = model_data['features']
    medians = model_data['medians']
    
    # Impute missing raw features using training medians, then derive engineered ones
    for col in RAW_ISOLATED_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer(df)

    # Resolve upstream chained risks (fall back to medians if not chained),
    # then derive the interaction terms that depend on them.
    for col in UPSTREAM_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer_chained(df)

    # Any remaining feature falls back to its training median too.
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
        'hypertension_risk': probs
    })
    return res

if __name__ == "__main__":
    train_and_evaluate()
