"""
Build a large, both-sex CKD dataset from NHANES to replace the 400-row UCI set.

WHY: the UCI CKD data (400 rows) shows near-perfect separation on its full lab
panel — an over-optimistic, tiny, single-source set. NHANES gives thousands of
adults of both sexes with real kidney labs, so we can define CKD by clinical
criteria and train an honest *checkup-safe* screening model (no leaky lab
features).

HOW CKD IS DEFINED (KDIGO markers, cross-sectional):
    eGFR is computed from serum creatinine with the race-free CKD-EPI 2021
    creatinine equation. A participant is labelled CKD-positive if
        eGFR < 60 mL/min/1.73m^2   OR   urine albumin/creatinine ratio >= 30 mg/g.
    (NHANES is a single visit, so this flags CKD *markers*, not the 3-month
    persistence required for a clinical diagnosis — documented as a caveat.)

DATA ACCESS: NHANES XPT files live at (current CDC layout)
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<begin-year>/DataFiles/<FILE>.xpt
This script tries to download them; if the network blocks cdc.gov (as some
sandboxes do), it falls back to reading local copies you place under
    data/raw/nhanes/<cycle>/<FILE>.XPT

To fetch the 2017-2018 cycle on a machine with internet, run (note CDC's current
URL layout: .../Public/<begin-year>/DataFiles/<FILE>.xpt):
    mkdir -p data/raw/nhanes/2017-2018 && cd data/raw/nhanes/2017-2018
    for f in DEMO_J BMX_J BPX_J BIOPRO_J TCHOL_J GLU_J SMQ_J ALB_CR_J; do
        curl -sSLO "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/$f.xpt"; done

Then:  python scripts/build_ckd_nhanes.py         # writes data/processed/ckd_nhanes_clean.csv
Self-test the eGFR math (no data needed):  python scripts/build_ckd_nhanes.py --selftest
"""
import os
import sys
import io
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_NHANES = os.path.join(ROOT, "data", "raw", "nhanes")
OUT = os.path.join(ROOT, "data", "processed", "ckd_nhanes_clean.csv")

# Cycle -> file suffix. 2017-2018 uses the "_J" suffix. Add more cycles here to
# pool a larger sample (the columns line up across recent cycles).
CYCLES = {"2017-2018": "_J"}

# Logical module name -> base file stem (suffix appended per cycle).
MODULES = {
    "DEMO": "DEMO",      # demographics: age, sex
    "BMX": "BMX",        # body measures: BMI, waist circumference
    "BPX": "BPX",        # blood pressure, resting pulse
    "BIOPRO": "BIOPRO",  # standard biochemistry: serum creatinine, uric acid
    "TCHOL": "TCHOL",    # total cholesterol
    "GLU": "GLU",        # fasting plasma glucose (fasting subsample)
    "SMQ": "SMQ",        # smoking questionnaire
    "ALB_CR": "ALB_CR",  # urine albumin / creatinine ratio
}


# ─────────────────────────── core, unit-testable transforms ───────────────────────────
def egfr_ckd_epi_2021(scr_mg_dl, age_years, is_female):
    """Race-free CKD-EPI 2021 creatinine eGFR (mL/min/1.73m^2).

    eGFR = 142 * min(Scr/k,1)^a * max(Scr/k,1)^-1.200 * 0.9938^age * (1.012 if female)
    with k = 0.7 (female) / 0.9 (male) and a = -0.241 (female) / -0.302 (male).
    Accepts scalars or numpy arrays.
    """
    scr = np.asarray(scr_mg_dl, dtype=float)
    age = np.asarray(age_years, dtype=float)
    female = np.asarray(is_female, dtype=bool)

    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    egfr = (142.0
            * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** (-1.200)
            * 0.9938 ** age
            * np.where(female, 1.012, 1.0))
    return egfr


def ckd_label(egfr, acr_mg_g):
    """CKD marker positive if eGFR < 60 OR ACR >= 30 (mg/g). NaN-safe: a criterion
    that is missing simply does not trigger; a row positive on either known
    criterion is positive."""
    egfr = np.asarray(egfr, dtype=float)
    acr = np.asarray(acr_mg_g, dtype=float)
    low_egfr = np.where(np.isnan(egfr), False, egfr < 60.0)
    albuminuria = np.where(np.isnan(acr), False, acr >= 30.0)
    return (low_egfr | albuminuria).astype(int)


def avg_bp(df, cols):
    """Row-wise mean of the available (non-null, >0) BP readings."""
    sub = df[cols].where(df[cols] > 0)
    return sub.mean(axis=1, skipna=True)


def derive_smoking(smq020, smq040):
    """Binary current-smoker flag. SMQ040 (now smoke) 1/2 = every day / some days
    -> current smoker. Otherwise non-current. SMQ040 is only asked of ever-smokers
    (SMQ020==1); everyone else is a non-smoker."""
    smq040 = pd.to_numeric(smq040, errors="coerce")
    return smq040.isin([1.0, 2.0]).astype(float)


# ─────────────────────────── IO ───────────────────────────
def _read_xpt(cycle, suffix, stem):
    """Read one NHANES module as a DataFrame: local copy first, then CDC.

    Accepts a local file in either case (``BIOPRO_J.XPT`` or ``.xpt``). The CDC
    download URL follows the current layout:
    ``.../Nchs/Data/Nhanes/Public/<begin-year>/DataFiles/<FILE>.xpt``.
    """
    base = f"{stem}{suffix}"
    for ext in (".XPT", ".xpt"):
        local = os.path.join(RAW_NHANES, cycle, base + ext)
        if os.path.exists(local):
            return pd.read_sas(local, format="xport")
    begin_year = cycle.split("-")[0]
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{begin_year}/DataFiles/{base}.xpt"
    fname = base + ".xpt"
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as resp:
            return pd.read_sas(io.BytesIO(resp.read()), format="xport")
    except Exception as e:
        raise FileNotFoundError(
            f"Could not read {fname}: no local copy at {local} and download failed "
            f"({type(e).__name__}: {e}). See this script's docstring for fetch steps."
        )


def build():
    frames = []
    for cycle, suffix in CYCLES.items():
        mods = {name: _read_xpt(cycle, suffix, stem) for name, stem in MODULES.items()}

        demo = mods["DEMO"][["SEQN", "RIDAGEYR", "RIAGENDR"]]
        m = demo.copy()
        m = m.merge(mods["BMX"][["SEQN", "BMXBMI", "BMXWAIST", "BMXHT"]], on="SEQN", how="left")
        bpx = mods["BPX"]
        m = m.merge(bpx[["SEQN", "BPXPLS"] + [c for c in bpx.columns
                                    if c.startswith(("BPXSY", "BPXDI"))]], on="SEQN", how="left")
        m = m.merge(mods["BIOPRO"][["SEQN", "LBXSCR", "LBXSUA", "LBXSBU", "LBXSTR"]], on="SEQN", how="left")
        m = m.merge(mods["TCHOL"][["SEQN", "LBXTC"]], on="SEQN", how="left")
        m = m.merge(mods["GLU"][["SEQN", "LBXGLU"]], on="SEQN", how="left")
        m = m.merge(mods["SMQ"][["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left")
        m = m.merge(mods["ALB_CR"][["SEQN", "URDACT"]], on="SEQN", how="left")
        m["_cycle"] = cycle
        frames.append(m)

    df = pd.concat(frames, ignore_index=True)

    # Adults with a serum creatinine measurement (needed for eGFR).
    df = df[(df["RIDAGEYR"] >= 18) & df["LBXSCR"].notna()].copy().reset_index(drop=True)

    is_female = df["RIAGENDR"] == 2
    df["egfr"] = egfr_ckd_epi_2021(df["LBXSCR"], df["RIDAGEYR"], is_female)
    df["classification"] = ckd_label(df["egfr"], df["URDACT"])

    sy = [c for c in df.columns if c.startswith("BPXSY")]
    di = [c for c in df.columns if c.startswith("BPXDI")]

    out = pd.DataFrame()
    out["patient_id"] = "nhanes_" + df["SEQN"].astype(int).astype(str)
    out["age"] = df["RIDAGEYR"].astype(float)
    out["sex"] = np.where(is_female, 0.0, 1.0)
    out["bmi"] = df["BMXBMI"].astype(float)
    out["systolic_bp"] = avg_bp(df, sy)
    out["diastolic_bp"] = avg_bp(df, di)
    out["glucose"] = df["LBXGLU"].astype(float)
    out["cholesterol"] = df["LBXTC"].astype(float)
    out["smoking"] = derive_smoking(df["SMQ020"], df["SMQ040"])
    out["alcohol"] = np.nan
    out["physical_activity"] = np.nan
    out["family_history"] = np.nan
    out["waist_circumference"] = df["BMXWAIST"].astype(float)
    out["resting_pulse"] = avg_bp(df, ["BPXPLS"])
    out["uric_acid"] = df["LBXSUA"].astype(float)
    out["height"] = df["BMXHT"].astype(float)
    out["bun"] = df["LBXSBU"].astype(float)
    out["triglycerides"] = df["LBXSTR"].astype(float)
    # Lab extras (diagnostic — NOT for the checkup-safe model; they define CKD).
    out["serum_creatinine"] = df["LBXSCR"].astype(float)
    out["egfr"] = df["egfr"].astype(float)
    out["acr"] = df["URDACT"].astype(float)
    out["classification"] = df["classification"].astype(int)

    # Median-impute the checkup feature columns so the training pipeline (SMOTE)
    # has clean input — mirrors how the other processed datasets are cleaned.
    # Fasting glucose is a NHANES subsample, so it is the most-imputed column.
    for col in ["bmi", "systolic_bp", "diastolic_bp", "glucose", "cholesterol", "smoking",
                "waist_circumference", "resting_pulse", "uric_acid", "height", "bun", "triglycerides"]:
        if out[col].isnull().any():
            med = out[col].median()
            out[col] = out[col].fillna(med if pd.notna(med) else 0.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    prev = out["classification"].mean()
    print(f"Wrote {OUT}")
    print(f"  rows: {len(out)}  |  CKD-positive: {prev:.1%}")
    print(f"  sex: {out['sex'].value_counts().to_dict()} (0=female, 1=male)")
    print(f"  eGFR  median {out['egfr'].median():.1f}  (<60: {(out['egfr'] < 60).mean():.1%})")
    print(f"  ACR>=30: {(out['acr'] >= 30).mean():.1%}")
    return out


# ─────────────────────────── self-test (no data required) ───────────────────────────
def _selftest():
    # Reference eGFR values from the NKF CKD-EPI 2021 calculator (±0.6 tolerance).
    cases = [
        # (scr, age, female, expected)
        (0.9, 50, True, 77.9),
        (1.2, 60, False, 69.2),
        (0.8, 40, True, 95.4),
        (1.5, 70, False, 49.8),
    ]
    ok = True
    for scr, age, fem, exp in cases:
        got = float(egfr_ckd_epi_2021(scr, age, fem))
        flag = "OK " if abs(got - exp) < 0.6 else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  [{flag}] eGFR(scr={scr}, age={age}, female={fem}) = {got:.1f} (expected ~{exp})")

    # Label logic
    lbl = ckd_label(np.array([80.0, 55.0, 90.0, np.nan]),
                    np.array([10.0, 5.0, 45.0, 40.0]))
    assert lbl.tolist() == [0, 1, 1, 1], lbl.tolist()
    print(f"  [OK ] ckd_label -> {lbl.tolist()} (expected [0, 1, 1, 1])")

    # BP averaging ignores nulls / zeros
    dfb = pd.DataFrame({"BPXSY1": [120, 0], "BPXSY2": [130, 140], "BPXSY3": [np.nan, 138]})
    got_bp = avg_bp(dfb, ["BPXSY1", "BPXSY2", "BPXSY3"]).round(1).tolist()
    assert got_bp == [125.0, 139.0], got_bp
    print(f"  [OK ] avg_bp -> {got_bp} (expected [125.0, 139.0])")

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    build()
