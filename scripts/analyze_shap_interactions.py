"""
SHAP interaction analysis: find which feature *pairs* actually interact in
each shipped model, rather than guessing engineered interaction terms by
clinical intuition (which is how pulse_pressure/age_glucose_interaction/
bmi_age_interaction were originally picked).

Uses XGBoost's TreeExplainer.shap_interaction_values, which decomposes each
prediction into per-feature main effects plus pairwise interaction effects.
Ranks feature pairs by mean |interaction SHAP value| across a sample of the
training data — the pairs at the top are where the model's prediction for
one feature genuinely depends on the value of another, not just two features
that are each independently predictive.

Usage:
    python scripts/analyze_shap_interactions.py                # all 4 models
    python scripts/analyze_shap_interactions.py --model diabetes
    python scripts/analyze_shap_interactions.py --top 15 --sample 2000
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import shap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models import diabetes_model, ckd_model, heart_model, hypertension_model
from src.models.upstream import add_upstream_risks
from src.models.tuned_params import load_tuned

DATA = os.path.join(ROOT, "data", "processed")


def _load_diabetes():
    df = pd.read_csv(os.path.join(DATA, "diabetes_nhanes_clean.csv"))
    df = diabetes_model._engineer(df)
    return df[diabetes_model.CHECKUP_SAFE_FEATURES], df["Outcome"], diabetes_model.XGB_PARAMS


def _load_ckd():
    df = pd.read_csv(os.path.join(DATA, "ckd_nhanes_clean.csv"))
    df = ckd_model._engineer(df)
    feats = ckd_model.RAW_CHECKUP_FEATURES + ckd_model.ENGINEERED_FEATURES
    params = {
        **load_tuned("ckd", {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}),
        "random_state": 42, "eval_metric": "logloss",
        "monotone_constraints": ckd_model.SAFE_MONOTONIC,
    }
    return df[feats], df["classification"], params


def _load_heart():
    df = pd.read_csv(os.path.join(DATA, "heart_clean.csv"))
    df = heart_model._engineer(df)
    df = add_upstream_risks(df)
    df = heart_model._engineer_chained(df)
    feats = heart_model.RAW_ISOLATED_FEATURES + heart_model.ENGINEERED_FEATURES + ["diabetes_risk", "ckd_risk"] + heart_model.CHAINED_ENGINEERED_FEATURES
    chained_params = {**heart_model.XGB_PARAMS, "monotone_constraints": heart_model.CHAINED_MONOTONIC}
    return df[feats], df["target"], chained_params


def _load_hypertension():
    df = pd.read_csv(os.path.join(DATA, "hypertension_nhanes_clean.csv"))
    df = hypertension_model._engineer(df)
    df = add_upstream_risks(df)
    df = hypertension_model._engineer_chained(df)
    feats = hypertension_model.RAW_ISOLATED_FEATURES + hypertension_model.ENGINEERED_FEATURES + ["diabetes_risk", "ckd_risk"] + hypertension_model.CHAINED_ENGINEERED_FEATURES
    chained_params = {**hypertension_model.XGB_PARAMS, "monotone_constraints": hypertension_model.CHAINED_MONOTONIC}
    return df[feats], df["Outcome"], chained_params


LOADERS = {
    "diabetes": _load_diabetes,
    "ckd": _load_ckd,
    "heart": _load_heart,
    "hypertension": _load_hypertension,
}


def analyze_one(name, top_n, sample_size, seed=42):
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    X, y, params = LOADERS[name]()

    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X, y)
    model = XGBClassifier(**params)
    model.fit(X_res, y_res)

    sample = X.sample(n=min(sample_size, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    interactions = explainer.shap_interaction_values(sample)  # (n, f, f)

    mean_abs = np.abs(interactions).mean(axis=0)
    feats = list(X.columns)
    pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            pairs.append((feats[i], feats[j], mean_abs[i, j]))
    pairs.sort(key=lambda p: p[2], reverse=True)

    print(f"\n{'='*60}\n{name} — top {top_n} feature-pair interactions\n{'='*60}")
    for f1, f2, score in pairs[:top_n]:
        print(f"  {f1:26s} x {f2:26s}  {score:.5f}")

    print(f"\n  Main-effect magnitudes (diagonal), for context:")
    main = sorted(zip(feats, np.diag(mean_abs)), key=lambda p: p[1], reverse=True)
    for f, score in main[:8]:
        print(f"    {f:26s}  {score:.5f}")

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(LOADERS.keys()), default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--sample", type=int, default=1500)
    args = parser.parse_args()

    models = [args.model] if args.model else list(LOADERS.keys())
    for name in models:
        analyze_one(name, args.top, args.sample)


if __name__ == "__main__":
    main()
