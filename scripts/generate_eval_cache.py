"""
Generates ROC curve data and calibration curve data for all 4 models.
Uses the saved checkup-safe models against a held-out 20% test split (stratified).
Writes results to models/eval_cache.pkl so the dashboard can load them instantly.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS, "eval_cache.pkl")

# ── Feature schemas (must match what each model was trained on) ──
SCHEMAS = {
    "Diabetes": {
        "csv":    "diabetes_clean.csv",
        "pkl":    "diabetes_model.pkl",
        "target": "Outcome",
        "feats":  None,   # loaded from pkl
    },
    "CKD": {
        "csv":    "ckd_clean.csv",
        "pkl":    "ckd_model.pkl",
        "target": "classification",
        "feats":  None,
    },
    "Heart Disease": {
        "csv":    "heart_clean.csv",
        "pkl":    "heart_model.pkl",
        "target": "target",
        "feats":  None,
    },
    "Hypertension": {
        "csv":    "hypertension_clean.csv",
        "pkl":    "hypertension_model.pkl",
        "target": "cardio",
        "feats":  None,
    },
}

cache = {}

for name, cfg in SCHEMAS.items():
    print(f"\n── {name} ──")
    df = pd.read_csv(os.path.join(DATA, cfg["csv"]))
    model_data = joblib.load(os.path.join(MODELS, cfg["pkl"]))
    model  = model_data["model"]
    feats  = model_data["features"]
    target = cfg["target"]

    # Find target column (case-insensitive fallback)
    if target not in df.columns:
        candidates = [c for c in df.columns if c.lower() == target.lower()]
        if not candidates:
            print(f"  WARNING: target '{target}' not found. Available: {list(df.columns[:10])}")
            continue
        target = candidates[0]

    # Keep only rows with all needed columns
    needed = feats + [target]
    df = df[[c for c in needed if c in df.columns]].dropna()

    # Align missing feature columns
    for f in feats:
        if f not in df.columns:
            df[f] = df[feats].median()[feats.index(f)] if f in feats else 0

    X = df[feats]
    y = df[target].astype(int)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    probs = model.predict_proba(X_test)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    print(f"  ROC AUC = {roc_auc:.4f}")

    # Calibration (fraction of positives vs mean predicted prob)
    n_bins = 10
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=n_bins, strategy="uniform")

    cache[name] = {
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
        "roc_auc":   roc_auc,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "n_test":    len(y_test),
        "pos_rate":  float(y_test.mean()),
    }

joblib.dump(cache, OUT)
print(f"\n✓ eval_cache.pkl saved to {OUT}")
print(f"  Keys: {list(cache.keys())}")
