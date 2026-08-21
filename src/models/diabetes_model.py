"""
Diabetes risk model (upstream).

Training cohort: NHANES 2021-2023 (7,912 adults, both sexes, ~15.5% positive),
built by ``scripts/build_diabetes_nhanes.py``.  This replaces the synthetic
Kaggle 100k dataset whose glucose column contained only 18 discrete values,
producing non-monotonic tree artifacts (e.g. glucose=158 predicting 4% risk).
NHANES glucose has 198 unique real clinical measurements.

Checkup-safe feature set (14 features): the original 8 (age, sex, bmi,
systolic_bp, diastolic_bp, glucose, cholesterol, smoking) plus 4 additional
NHANES measurements (waist_circumference, resting_pulse, uric_acid,
cigs_per_day) and 2 engineered features (pulse_pressure,
age_glucose_interaction) — see src/models/feature_engineering.py.

Monotonic constraints enforce clinically correct feature directions:
age, bmi, systolic_bp, glucose, waist_circumference, resting_pulse,
uric_acid, cigs_per_day, pulse_pressure, age_glucose_interaction are all
+1 (higher always increases predicted risk).  sex, diastolic_bp,
cholesterol, smoking are unconstrained (0) because their relationship with
diabetes risk is non-directional or context-dependent.
"""
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from src.models.feature_engineering import add_pulse_pressure, add_age_glucose_interaction

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_FEATURES = [
    'age', 'sex', 'bmi', 'systolic_bp', 'diastolic_bp',
    'glucose', 'cholesterol', 'smoking',
    'waist_circumference', 'resting_pulse', 'uric_acid', 'cigs_per_day',
]
ENGINEERED_FEATURES = ['pulse_pressure', 'age_glucose_interaction']
CHECKUP_SAFE_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_pulse_pressure(df)
    df = add_age_glucose_interaction(df)
    return df


# age↑ sex(any) bmi↑ sysBP↑ diaBP(any) glucose↑ chol(any) smoking(any)
# waist↑ pulse↑ uric_acid↑ cigs↑ pulse_pressure↑ age_glucose↑
MONOTONIC_CONSTRAINTS = (1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1)

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'monotone_constraints': MONOTONIC_CONSTRAINTS,
}


def train_and_evaluate():
    data_path = os.path.join(DATA_PROCESSED, "diabetes_nhanes_clean.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"NHANES diabetes data not found at {data_path}. "
            "Run scripts/build_diabetes_nhanes.py first."
        )
    df = pd.read_csv(data_path)
    df = _engineer(df)

    target_col = 'Outcome'
    X = df[CHECKUP_SAFE_FEATURES]
    y = df[target_col]

    medians = X.median().to_dict()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, aucs, f1s, confs = [], [], [], []

    print("Training Diabetes (NHANES) checkup-safe model (5-fold CV)...")
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train_res, y_train_res)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, preds))
        aucs.append(roc_auc_score(y_val, probs))
        f1s.append(f1_score(y_val, preds))
        confs.append(confusion_matrix(y_val, preds))

    avg_acc = np.mean(accs)
    avg_auc = np.mean(aucs)
    avg_f1 = np.mean(f1s)
    sum_conf = np.sum(confs, axis=0)

    print(f"\n=== Diabetes Checkup-Safe (NHANES) Performance ===")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"ROC AUC:  {avg_auc:.4f}")
    print(f"F1-Score: {avg_f1:.4f}")
    print("Confusion Matrix:\n", sum_conf)

    # Train final model on entire dataset
    print("\nTraining final model on entire dataset...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_res, y_res)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "diabetes_model.pkl")

    model_data = {
        'model': final_model,
        'features': CHECKUP_SAFE_FEATURES,
        'medians': medians,
    }
    joblib.dump(model_data, model_path)
    print(f"Saved checkup-safe model to {model_path}")

    return {'checkup_safe': (avg_acc, avg_auc, avg_f1, sum_conf)}


# --- HANDOFF PREDICTION INTERFACE ---

def _load_model_data():
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "diabetes_model.pkl")
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained diabetes model file not found at {model_path}. Run training first.")
    return joblib.load(model_path)

def predict_risk(patient_df) -> float:
    if isinstance(patient_df, dict):
        df = pd.DataFrame([patient_df])
    elif isinstance(patient_df, pd.Series):
        df = pd.DataFrame([patient_df])
    else:
        df = patient_df.copy()

    model_data = _load_model_data()
    model = model_data['model']
    features = model_data['features']
    medians = model_data['medians']

    for col in RAW_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer(df)

    df_feats = df[features]
    probs = model.predict_proba(df_feats)
    return float(probs[0, 1])

def predict_risk_batch(patient_df) -> pd.DataFrame:
    df = patient_df.copy()

    model_data = _load_model_data()
    model = model_data['model']
    features = model_data['features']
    medians = model_data['medians']

    for col in RAW_FEATURES:
        if col not in df.columns:
            df[col] = medians[col]
        else:
            df[col] = df[col].fillna(medians[col])
    df = _engineer(df)

    df_feats = df[features]
    probs = model.predict_proba(df_feats)[:, 1]

    res = pd.DataFrame({
        'patient_id': df['patient_id'] if 'patient_id' in df.columns else range(len(df)),
        'diabetes_risk': probs
    })
    return res

if __name__ == "__main__":
    train_and_evaluate()
