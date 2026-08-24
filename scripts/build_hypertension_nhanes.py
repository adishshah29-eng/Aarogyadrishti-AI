"""
Build the hypertension training cohort from NHANES 2021-2023.

This replaces the cardio/Kaggle dataset (70k rows), whose target ('cardio') is
a broad self-reported cardiovascular-disease flag rather than hypertension
specifically, and whose cholesterol/glucose are ordinal categories (1-3) rather
than continuous lab values. It also had no waist circumference, pulse, or
uric acid — this build adds all three from NHANES files already on disk.

Hypertension label (clinically standard, matches the ground truth used in
scripts/run_external_validation.py):
    Doctor-diagnosed high blood pressure (BPQ020 == 1)
Negative:
    BPQ020 == 2 (doctor says no)

NOTE: this cohort has no NHANES alcohol-use or physical-activity module
downloaded (ALQ/PAQ; blocked by this environment's egress policy — see
scripts/build_diabetes_nhanes.py's sibling note). Those two features are
therefore dropped from the checkup-safe set for this model, rather than kept
as unused ("dead") inputs collected from the user but never consumed by any
shipped model.

Requires the NHANES 2021-2023 XPT files in data/raw/nhanes/2021-2023/.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts.build_ckd_nhanes import avg_bp, derive_smoking

RAW = os.path.join(ROOT, "data", "raw", "nhanes", "2021-2023")
OUT = os.path.join(ROOT, "data", "processed", "hypertension_nhanes_clean.csv")


def _read(name):
    return pd.read_sas(os.path.join(RAW, f"{name}.xpt"), format="xport")


def main():
    demo = _read("DEMO_L")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    df = demo[demo["RIDAGEYR"] >= 18].copy().reset_index(drop=True)

    df = df.merge(_read("BMX_L")[["SEQN", "BMXBMI", "BMXWAIST", "BMXHT"]], on="SEQN", how="left")

    bpxo = _read("BPXO_L")
    sy = [c for c in bpxo.columns if c.startswith("BPXOSY")]
    di = [c for c in bpxo.columns if c.startswith("BPXODI")]
    pls = [c for c in bpxo.columns if c.startswith("BPXOPLS")]
    df = df.merge(bpxo[["SEQN"] + sy + di + pls], on="SEQN", how="left")

    df = df.merge(_read("TCHOL_L")[["SEQN", "LBXTC"]], on="SEQN", how="left")
    df = df.merge(_read("GLU_L")[["SEQN", "LBXGLU"]], on="SEQN", how="left")
    df = df.merge(_read("SMQ_L")[["SEQN", "SMQ020", "SMQ040", "SMD650"]], on="SEQN", how="left")
    df = df.merge(_read("BIOPRO_L")[["SEQN", "LBXSUA"]], on="SEQN", how="left")
    df = df.merge(_read("BPQ_L")[["SEQN", "BPQ020"]], on="SEQN", how="left")

    is_female = df["RIAGENDR"] == 2

    # ── Hypertension label ──
    bpq = df["BPQ020"]
    label = pd.Series(np.nan, index=df.index)
    label[bpq == 1] = 1
    label[bpq == 2] = 0

    # ── Canonical checkup features ──
    out = pd.DataFrame()
    out["patient_id"] = "nhanes21_ht_" + df["SEQN"].astype(int).astype(str)
    out["age"] = df["RIDAGEYR"].astype(float)
    out["sex"] = np.where(is_female, 0.0, 1.0)
    out["bmi"] = df["BMXBMI"].astype(float)
    out["systolic_bp"] = avg_bp(df, sy)
    out["diastolic_bp"] = avg_bp(df, di)
    out["glucose"] = df["LBXGLU"].astype(float)
    out["cholesterol"] = df["LBXTC"].astype(float)
    out["smoking"] = derive_smoking(df["SMQ020"], df["SMQ040"])
    out["waist_circumference"] = df["BMXWAIST"].astype(float)
    out["height"] = df["BMXHT"].astype(float)
    out["resting_pulse"] = avg_bp(df, pls)
    out["uric_acid"] = df["LBXSUA"].astype(float)
    out["cigs_per_day"] = pd.to_numeric(df["SMD650"], errors="coerce")
    out.loc[out["smoking"] == 0.0, "cigs_per_day"] = out.loc[out["smoking"] == 0.0, "cigs_per_day"].fillna(0.0)
    out["Outcome"] = label

    out = out.dropna(subset=["Outcome"]).reset_index(drop=True)
    out["Outcome"] = out["Outcome"].astype(int)

    feat_cols = ["age", "sex", "bmi", "systolic_bp", "diastolic_bp", "glucose", "cholesterol",
                 "smoking", "waist_circumference", "height", "resting_pulse", "uric_acid", "cigs_per_day"]
    for col in feat_cols:
        med = out[col].median()
        out[col] = out[col].fillna(med)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    n_pos = out["Outcome"].sum()
    n_neg = len(out) - n_pos
    print(f"Saved {OUT}")
    print(f"  {len(out)} adults  |  {n_pos} hypertensive ({n_pos/len(out):.1%})  |  {n_neg} not")
    print(f"  Features: {feat_cols}")


if __name__ == "__main__":
    main()
