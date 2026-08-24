"""
Optuna hyperparameter search for all four models.

Tunes XGBoost's capacity/regularization knobs (n_estimators, max_depth,
learning_rate, min_child_weight, subsample, colsample_bytree, reg_alpha,
reg_lambda) against 5-fold CV mean ROC AUC, with SMOTE applied on the
training fold only — the exact same procedure each model's own
train_and_evaluate() uses, so a trial's score is directly comparable to the
reported CV numbers.

What's deliberately NOT tuned: monotonic constraints. Those encode a
clinical correctness requirement (age/BMI/glucose/... can never lower
predicted risk), not a performance knob, so they stay fixed at each model's
existing values regardless of what the search finds.

Usage:
    python scripts/tune_hyperparameters.py                  # all 4 models, 60 trials each
    python scripts/tune_hyperparameters.py --model diabetes --trials 100
    python scripts/tune_hyperparameters.py --trials 30       # faster, all models

Writes configs/tuned_hyperparams.json. Each model's XGB_PARAMS reads this
file at import time and falls back to its hardcoded defaults if a model's
entry (or the file) is absent, so tuning is opt-in and reproducible: delete
the file (or a model's key in it) to revert to the hand-picked defaults.
"""
import argparse
import json
import os
import sys

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models import diabetes_model, ckd_model, heart_model, hypertension_model
from src.models.upstream import add_upstream_risks

optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG_PATH = os.path.join(ROOT, "configs", "tuned_hyperparams.json")

TUNABLE_SPACE = {
    "n_estimators":      lambda t: t.suggest_int("n_estimators", 80, 400),
    "max_depth":         lambda t: t.suggest_int("max_depth", 3, 7),
    "learning_rate":     lambda t: t.suggest_float("learning_rate", 0.01, 0.2, log=True),
    "min_child_weight":  lambda t: t.suggest_int("min_child_weight", 1, 10),
    "subsample":         lambda t: t.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree":  lambda t: t.suggest_float("colsample_bytree", 0.6, 1.0),
    "reg_alpha":         lambda t: t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    "reg_lambda":        lambda t: t.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
}


def _load_diabetes():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "diabetes_nhanes_clean.csv"))
    df = diabetes_model._engineer(df)
    X = df[diabetes_model.CHECKUP_SAFE_FEATURES]
    y = df["Outcome"]
    return X, y, diabetes_model.MONOTONIC_CONSTRAINTS


def _load_ckd():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "ckd_nhanes_clean.csv"))
    df = ckd_model._engineer(df)
    feats = ckd_model.RAW_CHECKUP_FEATURES + ckd_model.ENGINEERED_FEATURES
    X = df[feats]
    y = df["classification"]
    return X, y, ckd_model.SAFE_MONOTONIC


def _load_heart():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "heart_clean.csv"))
    df = heart_model._engineer(df)
    df = add_upstream_risks(df)
    df = heart_model._engineer_chained(df)
    feats = heart_model.RAW_ISOLATED_FEATURES + heart_model.ENGINEERED_FEATURES + ["diabetes_risk", "ckd_risk"] + heart_model.CHAINED_ENGINEERED_FEATURES
    X = df[feats]
    y = df["target"]
    return X, y, heart_model.CHAINED_MONOTONIC


def _load_hypertension():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "hypertension_nhanes_clean.csv"))
    df = hypertension_model._engineer(df)
    df = add_upstream_risks(df)
    df = hypertension_model._engineer_chained(df)
    feats = hypertension_model.RAW_ISOLATED_FEATURES + hypertension_model.ENGINEERED_FEATURES + ["diabetes_risk", "ckd_risk"] + hypertension_model.CHAINED_ENGINEERED_FEATURES
    X = df[feats]
    y = df["Outcome"]
    return X, y, hypertension_model.CHAINED_MONOTONIC


LOADERS = {
    "diabetes": _load_diabetes,
    "ckd": _load_ckd,
    "heart": _load_heart,
    "hypertension": _load_hypertension,
}


def _cv_auc(X, y, params, monotone_constraints, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        smote = SMOTE(random_state=seed)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            **params,
            monotone_constraints=monotone_constraints,
            random_state=seed,
            eval_metric="logloss",
        )
        model.fit(X_train_res, y_train_res)
        probs = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, probs))
    return float(np.mean(aucs))


def tune_one(name, n_trials, baseline_auc=None):
    print(f"\n{'='*60}\nTuning {name} ({n_trials} trials)\n{'='*60}")
    X, y, monotone = LOADERS[name]()

    def objective(trial):
        params = {k: fn(trial) for k, fn in TUNABLE_SPACE.items()}
        return _cv_auc(X, y, params, monotone)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_auc = study.best_value
    print(f"  Best CV AUC: {best_auc:.4f}" + (f"  (baseline: {baseline_auc:.4f}, "
          f"{'+' if best_auc >= baseline_auc else ''}{best_auc - baseline_auc:.4f})" if baseline_auc else ""))
    print(f"  Best params: {study.best_params}")
    return study.best_params, best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(LOADERS.keys()), default=None,
                         help="Tune a single model; omit to tune all four.")
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args()

    models = [args.model] if args.model else list(LOADERS.keys())

    existing = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            existing = json.load(f)

    for name in models:
        best_params, best_auc = tune_one(name, args.trials)
        existing[name] = {**best_params, "_cv_auc": round(best_auc, 4)}

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved tuned hyperparameters to {CONFIG_PATH}")
    print("Run scripts/retrain_all.py to retrain with these params.")


if __name__ == "__main__":
    main()
