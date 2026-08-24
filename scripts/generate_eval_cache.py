"""
Generate ROC and calibration curve data for all 4 models on a genuinely
held-out test split, and cache it to models/eval_cache.pkl for the dashboard.

Why retrain here?
    The shipped .pkl models are deliberately trained on 100% of their data
    (final_model.fit(...) in each model's train_and_evaluate). Evaluating that
    same model on a "held-out" slice of its own training data leaks — the model
    has already seen every one of those rows. To report an honest held-out
    curve we instead train a FRESH model with the identical configuration on the
    80% train split and score the untouched 20% test split. The curves therefore
    describe the generalization of the shipped *recipe*, not an in-sample fit.

For the downstream (chained) models, the upstream Diabetes/CKD risk features are
regenerated with src.models.upstream.add_upstream_risks so the evaluated feature
set matches what the model actually ships with. Each model's own feature-
engineering helper (pulse pressure, interaction terms, ...) is applied the same
way it is at training time, so the held-out feature set matches the shipped one.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models.upstream import add_upstream_risks, UPSTREAM_FEATURES
from src.models import diabetes_model, ckd_model, heart_model, hypertension_model
from src.models.tuned_params import load_tuned

DATA = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS, "eval_cache.pkl")

def _heart_engineer(df):
    return heart_model._engineer_chained(heart_model._engineer(df))


def _hypertension_engineer(df):
    return hypertension_model._engineer_chained(hypertension_model._engineer(df))


SCHEMAS = {
    "Diabetes":      {"csv": "diabetes_nhanes_clean.csv", "pkl": "diabetes_model.pkl",
                       "target": "Outcome",        "params": diabetes_model.XGB_PARAMS,
                       "engineer": diabetes_model._engineer},
    "CKD":           {"csv": "ckd_nhanes_clean.csv",      "pkl": "ckd_model.pkl",
                       "target": "classification", "params": {
                           **load_tuned('ckd', {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1}),
                           'random_state': 42, 'eval_metric': 'logloss',
                           'monotone_constraints': ckd_model.SAFE_MONOTONIC},
                       "engineer": ckd_model._engineer},
    "Heart Disease": {"csv": "heart_clean.csv",           "pkl": "heart_model.pkl",
                       "target": "target",         "params": {**heart_model.XGB_PARAMS, 'monotone_constraints': heart_model.CHAINED_MONOTONIC},
                       "engineer": _heart_engineer},
    "Hypertension":  {"csv": "hypertension_nhanes_clean.csv", "pkl": "hypertension_model.pkl",
                       "target": "Outcome",        "params": {**hypertension_model.XGB_PARAMS, 'monotone_constraints': hypertension_model.CHAINED_MONOTONIC},
                       "engineer": _hypertension_engineer},
}

cache = {}

for name, cfg in SCHEMAS.items():
    print(f"\n── {name} ──")
    df = pd.read_csv(os.path.join(DATA, cfg["csv"]))
    feats = joblib.load(os.path.join(MODELS, cfg["pkl"]))["features"]
    target = cfg["target"]

    # Downstream models chain on upstream risk scores that are not columns of
    # the raw processed CSV — regenerate them so the feature set matches ship.
    if any(f in UPSTREAM_FEATURES for f in feats) and not set(UPSTREAM_FEATURES).issubset(df.columns):
        df = add_upstream_risks(df)

    # Derive each model's engineered features (pulse pressure, interaction
    # terms, ...) the same way train_and_evaluate() does, so `feats` resolves.
    df = cfg["engineer"](df)

    if target not in df.columns:
        candidates = [c for c in df.columns if c.lower() == target.lower()]
        if not candidates:
            print(f"  WARNING: target '{target}' not found; skipping.")
            continue
        target = candidates[0]

    df = df[[c for c in feats + [target] if c in df.columns]].dropna()
    X = df[feats]
    y = df[target].astype(int)

    # Genuine held-out split: the model below never sees X_test / y_test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train a fresh model with the shipped configuration on the train split only.
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(**cfg["params"])
    model.fit(X_res, y_res)

    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    print(f"  Held-out ROC AUC = {roc_auc:.4f}  (n_test={len(y_test)})")

    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")

    cache[name] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": float(roc_auc),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "n_test": int(len(y_test)),
        "pos_rate": float(y_test.mean()),
    }

joblib.dump(cache, OUT)
print(f"\n✓ eval_cache.pkl saved to {OUT}")
print(f"  Keys: {list(cache.keys())}")
