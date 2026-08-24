"""
Contract tests for AarogyaDrishti AI.

These guard the classes of bug found in the audit so they cannot silently
regress:
  * unit/scale mismatches between the dashboard and a model (finding C1),
  * dead inputs that the UI collects but no model uses (finding H1),
  * the chaining wiring (downstream models must consume the upstream risks),
  * numeric correctness of the eGFR and CRI helpers.

Run:  python -m pytest tests/ -q      (or)   python tests/test_contracts.py
No network required; uses the shipped .pkl models and processed CSVs.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import joblib
from src.chaining.cri import get_full_risk_profile, compute_cri
from src.models.upstream import UPSTREAM_FEATURES, add_upstream_risks

MODELS = ["diabetes", "ckd", "heart", "hypertension"]
DATA = os.path.join(ROOT, "data", "processed")

# A realistic mid-range adult in canonical units (glucose/cholesterol in mg/dL).
# alcohol/physical_activity were dropped project-wide (no NHANES module for
# either is available in this environment, so no shipped model consumes them
# any more — see scripts/build_hypertension_nhanes.py); keeping them here
# would silently defeat test_no_dead_input_features below.
PATIENT = {
    "age": 55.0, "sex": 1.0, "bmi": 28.0, "systolic_bp": 135.0, "diastolic_bp": 85.0,
    "glucose": 140.0, "cholesterol": 220.0, "smoking": 1.0,
    "waist_circumference": 95.0, "resting_pulse": 78.0, "uric_acid": 5.5, "cigs_per_day": 0.0,
    "heartRate": 78.0, "cigsPerDay": 0.0, "prevalentHyp": 0.0, "BPMeds": 0.0,
}


def _load(name):
    return joblib.load(os.path.join(ROOT, "models", f"{name}_model.pkl"))


# ── C1: no model should receive an inference input outside its training range ──
def test_inference_inputs_within_training_ranges():
    """Every non-upstream feature the dashboard sends must fall inside the
    min/max the model was trained on. This is exactly the hypertension
    cholesterol/glucose scale bug (categorical 1-3 vs mg/dL)."""
    csvs = {"diabetes": "diabetes_nhanes_clean.csv", "ckd": "ckd_nhanes_clean.csv",
            "heart": "heart_clean.csv", "hypertension": "hypertension_nhanes_clean.csv"}
    problems = []
    for name in MODELS:
        feats = _load(name)["features"]
        df = pd.read_csv(os.path.join(DATA, csvs[name]))
        for f in feats:
            if f in UPSTREAM_FEATURES or f not in PATIENT:
                continue
            lo, hi = df[f].min(), df[f].max()
            v = PATIENT[f]
            # allow binary/degenerate columns; flag genuine scale mismatches
            if not (lo <= v <= hi) and hi > lo:
                problems.append(f"{name}.{f}: sends {v}, trained on [{lo}, {hi}]")
    assert not problems, "Input scale mismatch:\n  " + "\n  ".join(problems)


def test_cholesterol_moves_hypertension_score():
    """Cholesterol must actually influence the hypertension prediction across the
    clinical range (regression guard for the 'frozen cholesterol' bug)."""
    from src.models.hypertension_model import predict_risk as ht
    scores = []
    for chol in (160.0, 220.0, 300.0):
        d = dict(PATIENT, cholesterol=chol)
        scores.append(ht(add_upstream_risks(pd.DataFrame([d]))))
    assert max(scores) - min(scores) > 0.02, f"cholesterol inert: {scores}"


# ── H1: no dead inputs — every feature collected must reach at least one model ──
def test_no_dead_input_features():
    """Every feature in the encoded patient dict must be used by at least one
    shipped model (guards against collecting inputs nothing consumes)."""
    used = set()
    for name in MODELS:
        used |= set(_load(name)["features"])
    unused = [k for k in PATIENT if k not in used]
    assert not unused, f"inputs collected but used by no model: {unused}"


# ── Chaining: downstream models must actually consume the upstream risks ──
def test_downstream_models_are_chained():
    for name in ("heart", "hypertension"):
        feats = _load(name)["features"]
        assert set(UPSTREAM_FEATURES).issubset(feats), \
            f"{name} model is not chained; features={feats}"


def test_upstream_risk_changes_downstream_prediction():
    """Changing an upstream driver (glucose/BMI) must move the downstream scores
    through the chained features — proves chaining is live, not cosmetic."""
    high = get_full_risk_profile(dict(PATIENT, glucose=260.0, bmi=34.0))
    low = get_full_risk_profile(dict(PATIENT, glucose=85.0, bmi=22.0))
    assert abs(high["heart_risk"] - low["heart_risk"]) > 1e-6
    assert abs(high["hypertension_risk"] - low["hypertension_risk"]) > 1e-6


# ── CRI numeric correctness ──
def test_cri_matches_documented_formula():
    dia, ckd, heart, ht = 0.7, 0.1, 0.8, 0.1
    expected = (0.30 * dia + 0.20 * ckd + 0.25 * heart + 0.25 * ht
                + 0.15 * dia * heart + 0.10 * ht * heart + 0.05 * dia * ht)
    assert abs(compute_cri(dia, ckd, heart, ht) - expected) < 1e-9
    assert compute_cri(1.0, 1.0, 1.0, 1.0) <= 1.0  # clipped


# ── eGFR numeric correctness (the NHANES builder) ──
def test_egfr_reference_values():
    from scripts.build_ckd_nhanes import egfr_ckd_epi_2021
    assert abs(float(egfr_ckd_epi_2021(0.9, 50, True)) - 77.9) < 0.6
    assert abs(float(egfr_ckd_epi_2021(1.2, 60, False)) - 69.2) < 0.6


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
