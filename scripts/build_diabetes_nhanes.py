"""
Build the diabetes training cohort from NHANES 2021-2023.

This replaces the synthetic Kaggle 100k dataset (iammustafatz) whose glucose
column contains only 18 discrete values, producing non-monotonic tree artifacts
(e.g. glucose=158 predicting 4% diabetes risk).  NHANES glucose has 198 unique
values from real clinical measurements, eliminating the artifact entirely.

Diabetes label (clinically standard):
    HbA1c >= 6.5%  OR  doctor-diagnosed diabetes (DIQ010 == 1)
Negative:
    DIQ010 == 2 (doctor says no)  AND  HbA1c < 6.5 (or missing)

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
OUT = os.path.join(ROOT, "data", "processed", "diabetes_nhanes_clean.csv")


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
    df = df.merge(_read("GHB_L")[["SEQN", "LBXGH"]], on="SEQN", how="left")
    df = df.merge(_read("SMQ_L")[["SEQN", "SMQ020", "SMQ040", "SMD650"]], on="SEQN", how="left")
    df = df.merge(_read("DIQ_L")[["SEQN", "DIQ010"]], on="SEQN", how="left")
    df = df.merge(_read("BIOPRO_L")[["SEQN", "LBXSUA", "LBXSTR"]], on="SEQN", how="left")

    is_female = df["RIAGENDR"] == 2

    # ── Diabetes label ──
    ghb = df["LBXGH"]
    diq = df["DIQ010"]
    label = pd.Series(np.nan, index=df.index)
    label[(ghb >= 6.5) | (diq == 1)] = 1
    label[label.isna() & (diq == 2) & (ghb.isna() | (ghb < 6.5))] = 0

    # ── Canonical checkup features ──
    out = pd.DataFrame()
    out["patient_id"] = "nhanes21_" + df["SEQN"].astype(int).astype(str)
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
    out["triglycerides"] = df["LBXSTR"].astype(float)
    # Cigs/day is only asked of current smokers; non/former smokers -> 0.
    out["cigs_per_day"] = pd.to_numeric(df["SMD650"], errors="coerce")
    out.loc[out["smoking"] == 0.0, "cigs_per_day"] = out.loc[out["smoking"] == 0.0, "cigs_per_day"].fillna(0.0)
    out["Outcome"] = label

    out = out.dropna(subset=["Outcome"]).reset_index(drop=True)
    out["Outcome"] = out["Outcome"].astype(int)

    # Median-impute missing checkup features (glucose is a fasting subsample)
    feat_cols = ["age", "sex", "bmi", "systolic_bp", "diastolic_bp",
                 "glucose", "cholesterol", "smoking",
                 "waist_circumference", "resting_pulse", "uric_acid", "cigs_per_day",
                 "height", "triglycerides"]
    for col in feat_cols:
        med = out[col].median()
        out[col] = out[col].fillna(med)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    n_pos = out["Outcome"].sum()
    n_neg = len(out) - n_pos
    print(f"Saved {OUT}")
    print(f"  {len(out)} adults  |  {n_pos} diabetic ({n_pos/len(out):.1%})  |  {n_neg} non-diabetic")
    print(f"  Glucose unique values: {out['glucose'].nunique()}")
    print(f"  Features: {feat_cols}")


if __name__ == "__main__":
    main()
