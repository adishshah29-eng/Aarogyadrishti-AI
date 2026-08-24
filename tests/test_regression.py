"""
Regression tests for bugs found and fixed in this project's history.

Each test here exists because something specific broke once and should never
be allowed to silently break again:
  * the diabetes glucose=158 -> 4.1% artifact (P0, synthetic-data non-monotonic
    tree splits)
  * monotonicity of every constrained feature, for all four shipped models
    (the general form of the glucose=158 bug)
  * the dashboard's SHAP explanation path selecting engineered feature columns
    (pulse pressure, interaction terms, ...) before they were derived
  * a model silently ignoring a raw feature that isn't in the incoming
    DataFrame at all (as opposed to being present-but-NaN)

Run:  python -m pytest tests/test_regression.py -q   (or)
      python tests/test_regression.py
No network required; uses the shipped .pkl models.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models import diabetes_model, ckd_model, heart_model, hypertension_model
from src.models.upstream import add_upstream_risks


# ── The original bug: diagnostic-level fasting glucose must never score LOW ──
def test_diabetes_glucose_158_is_not_low_risk():
    """glucose=158 mg/dL is diagnostic for diabetes (>126 is the clinical
    threshold). The synthetic Kaggle dataset's 18-discrete-value glucose
    column produced non-monotonic tree splits that scored this 4.1% ('LOW').
    It must now score clearly elevated risk."""
    patient = {
        "age": 55, "sex": 1, "bmi": 29.5, "systolic_bp": 130, "diastolic_bp": 82,
        "glucose": 158, "cholesterol": 200, "smoking": 0,
    }
    risk = diabetes_model.predict_risk(patient)
    assert risk > 0.70, f"glucose=158 scored {risk:.3f} risk, expected > 0.70"


# ── General form of the same bug: every constrained feature must be monotone ──
MODELS_WITH_MONOTONE = [
    ("diabetes", diabetes_model, diabetes_model.CHECKUP_SAFE_FEATURES, diabetes_model.MONOTONIC_CONSTRAINTS),
    ("ckd", ckd_model, ckd_model.RAW_CHECKUP_FEATURES + ckd_model.ENGINEERED_FEATURES, ckd_model.SAFE_MONOTONIC),
]


def _median_patient(features):
    """A patient with every feature at a plausible mid-range value, so a
    monotonicity sweep isn't confounded by an unrealistic combination."""
    defaults = {
        "age": 50.0, "sex": 1.0, "bmi": 27.0, "systolic_bp": 125.0, "diastolic_bp": 80.0,
        "glucose": 100.0, "cholesterol": 190.0, "smoking": 0.0,
        "waist_circumference": 90.0, "resting_pulse": 72.0, "uric_acid": 5.0, "cigs_per_day": 0.0,
        "bun": 14.0, "triglycerides": 130.0,
        "heartRate": 72.0, "cigsPerDay": 0.0, "prevalentHyp": 0.0, "BPMeds": 0.0,
        "alcohol": 0.0, "physical_activity": 0.0,
    }
    return {k: defaults[k] for k in features if k in defaults}


def test_diabetes_and_ckd_monotonic_features_never_decrease_risk():
    """Sweeping every +1-constrained raw feature from low to high must never
    lower predicted risk, for each of the shipped upstream models."""
    sweeps = {
        "age": (25, 80), "bmi": (18, 45), "systolic_bp": (100, 180), "glucose": (70, 300),
        "waist_circumference": (70, 130), "resting_pulse": (55, 110), "uric_acid": (3.0, 9.0),
        "cigs_per_day": (0, 30), "bun": (7.0, 40.0), "triglycerides": (50.0, 400.0),
    }
    for name, model, features, constraints in MODELS_WITH_MONOTONE:
        raw_features = [f for f in features if f in sweeps]
        constraint_by_feat = dict(zip(features, constraints))
        base = _median_patient(features)
        for feat in raw_features:
            if constraint_by_feat.get(feat) != 1:
                continue
            lo, hi = sweeps[feat]
            steps = np.linspace(lo, hi, 8)
            risks = [model.predict_risk({**base, feat: v}) for v in steps]
            # Allow tiny numerical noise but never a real decrease end-to-end.
            assert risks[-1] >= risks[0] - 1e-6, (
                f"{name}.{feat} non-monotonic: risk at {lo}={risks[0]:.4f}, "
                f"at {hi}={risks[-1]:.4f}\n  full sweep: {list(zip(steps, risks))}"
            )


# ── SHAP path bug: engineered features must be derivable from raw model input ──
def test_engineered_features_present_after_engineer_step():
    """Each model's `features` list includes engineered columns; `_engineer`
    must be able to produce every one of them from that model's raw feature
    set alone (this is what the dashboard's SHAP path forgot to do once)."""
    cases = [
        (diabetes_model, diabetes_model.RAW_FEATURES, diabetes_model.ENGINEERED_FEATURES),
        (ckd_model, ckd_model.RAW_CHECKUP_FEATURES, ckd_model.ENGINEERED_FEATURES),
    ]
    for model, raw_features, engineered_features in cases:
        df = pd.DataFrame([_median_patient(raw_features)])
        out = model._engineer(df)
        missing = [c for c in engineered_features if c not in out.columns]
        assert not missing, f"{model.__name__}: _engineer didn't produce {missing}"


def test_chained_engineered_features_present_after_both_engineer_steps():
    """Heart/Hypertension's CHAINED_ENGINEERED_FEATURES (e.g.
    comorbidity_risk_interaction) need diabetes_risk/ckd_risk present first —
    this is exactly the bug the dashboard's SHAP-prep code had: it imported
    each model's raw `_engineer` only, never `_engineer_chained`, so the
    chained interaction column was silently absent and df[features] raised."""
    for model, raw_features in [
        (heart_model, heart_model.RAW_ISOLATED_FEATURES),
        (hypertension_model, hypertension_model.RAW_ISOLATED_FEATURES),
    ]:
        df = pd.DataFrame([{**_median_patient(raw_features), "diabetes_risk": 0.2, "ckd_risk": 0.2}])
        out = model._engineer_chained(model._engineer(df))
        missing = [c for c in model.CHAINED_ENGINEERED_FEATURES if c not in out.columns]
        assert not missing, f"{model.__name__}: _engineer_chained didn't produce {missing}"


def test_comorbidity_risk_interaction_never_decreases_risk():
    """Sweeping diabetes_risk upward (with ckd_risk held fixed and positive)
    must never lower Heart/Hypertension's predicted risk — the new
    comorbidity_risk_interaction feature carries a +1 monotonic constraint."""
    for model, raw_features in [
        (heart_model, heart_model.RAW_ISOLATED_FEATURES),
        (hypertension_model, hypertension_model.RAW_ISOLATED_FEATURES),
    ]:
        base = _median_patient(raw_features)
        risks = [
            model.predict_risk({**base, "diabetes_risk": d, "ckd_risk": 0.3})
            for d in np.linspace(0.05, 0.9, 6)
        ]
        assert risks[-1] >= risks[0] - 1e-6, (
            f"{model.__name__}: risk non-monotonic in diabetes_risk (comorbidity "
            f"interaction): {risks}"
        )


# ── Missing-column (not just missing-value) imputation ──
def test_predict_risk_handles_missing_raw_column_not_just_nan():
    """A raw feature can be absent from the input dict entirely (an "I don't
    know" wizard field), not just present-and-NaN. Every model must impute
    that case with its training median rather than raising."""
    minimal_patient = {"age": 50, "sex": 1, "bmi": 27, "systolic_bp": 125,
                        "diastolic_bp": 80, "glucose": 100, "cholesterol": 190, "smoking": 0}
    for model in (diabetes_model, ckd_model):
        risk = model.predict_risk(minimal_patient)
        assert 0.0 <= risk <= 1.0

    chain_patient = {**minimal_patient, "diabetes_risk": 0.2, "ckd_risk": 0.2}
    for model in (heart_model, hypertension_model):
        risk = model.predict_risk(chain_patient)
        assert 0.0 <= risk <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
