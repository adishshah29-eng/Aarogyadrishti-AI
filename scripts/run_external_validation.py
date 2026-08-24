"""
External validation: test the SHIPPED models on a cohort they have never
seen during training.

Why this matters: every metric reported elsewhere (5-fold CV, the 80/20
held-out split) comes from the SAME dataset each model was built on. That
proves the model fits its own distribution well, but not that it generalizes
to new patients. This script uses NHANES **August 2021-August 2023** — a
completely different, more recent survey wave, covering different people than
any training dataset (CKD: NHANES 2017-2018; Heart: Framingham) — as a
genuinely independent test cohort for CKD and Heart Disease.

NOTE on Diabetes and Hypertension: as of the P0 monotonic-constraints fix and
the P1 hypertension dataset swap, both the Diabetes and Hypertension models
now train directly on NHANES 2021-2023 (see scripts/build_diabetes_nhanes.py,
scripts/build_hypertension_nhanes.py) to eliminate synthetic-data artifacts
and a broad self-reported ('cardio') label respectively. That means THIS SAME
COHORT is no longer an independent test set for either — evaluating them here
would be in-sample, not external, validation. Both are therefore excluded
from this script's external-validation table; their only honest out-of-sample
check is the 5-fold CV reported in reports/baseline_metrics.md.

Ground-truth labels are derived independently of our models, from objective
lab values and doctor-diagnosis questionnaire answers, NOT from what our
models predict (that would be circular):
  - Heart disease: doctor-diagnosed coronary heart disease / angina /
                    heart attack (MCQ160C/D/E == 1)
  - CKD:           eGFR < 60 (CKD-EPI 2021) OR urine ACR >= 30 (KDIGO)
                    -- same clinical criteria used to build the CKD training
                    labels, but computed here on an entirely different cohort.

Usage:
    python scripts/run_external_validation.py
Requires the 12 raw NHANES files in data/raw/nhanes/2021-2023/ (see
docs/ckd_nhanes_upgrade.md for the download steps; same pattern, cycle 2021).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.build_ckd_nhanes import egfr_ckd_epi_2021, ckd_label, avg_bp, derive_smoking
from src.models import diabetes_model, ckd_model, heart_model
from src.chaining.cri import compute_cri

RAW = os.path.join(ROOT, "data", "raw", "nhanes", "2021-2023")
OUT_MD = os.path.join(ROOT, "reports", "external_validation.md")
OUT_CACHE = os.path.join(ROOT, "models", "external_eval_cache.pkl")


def _read(name):
    return pd.read_sas(os.path.join(RAW, f"{name}.xpt"), format="xport")


def build_cohort():
    demo = _read("DEMO_L")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    df = demo.copy()
    df = df.merge(_read("BMX_L")[["SEQN", "BMXBMI", "BMXWAIST", "BMXHT"]], on="SEQN", how="left")
    bpxo = _read("BPXO_L")
    sy = [c for c in bpxo.columns if c.startswith("BPXOSY")]
    di = [c for c in bpxo.columns if c.startswith("BPXODI")]
    pls = [c for c in bpxo.columns if c.startswith("BPXOPLS")]
    df = df.merge(bpxo[["SEQN"] + sy + di + pls], on="SEQN", how="left")
    df = df.merge(_read("BIOPRO_L")[["SEQN", "LBXSCR", "LBXSUA", "LBXSBU", "LBXSTR"]], on="SEQN", how="left")
    df = df.merge(_read("TCHOL_L")[["SEQN", "LBXTC"]], on="SEQN", how="left")
    df = df.merge(_read("GLU_L")[["SEQN", "LBXGLU"]], on="SEQN", how="left")
    df = df.merge(_read("GHB_L")[["SEQN", "LBXGH"]], on="SEQN", how="left")
    df = df.merge(_read("SMQ_L")[["SEQN", "SMQ020", "SMQ040", "SMD650"]], on="SEQN", how="left")
    df = df.merge(_read("ALB_CR_L")[["SEQN", "URDACT"]], on="SEQN", how="left")
    df = df.merge(_read("DIQ_L")[["SEQN", "DIQ010"]], on="SEQN", how="left")
    df = df.merge(_read("BPQ_L")[["SEQN", "BPQ020"]], on="SEQN", how="left")
    mcq = _read("MCQ_L")
    df = df.merge(mcq[["SEQN", "MCQ160C", "MCQ160D", "MCQ160E"]], on="SEQN", how="left")

    df = df[df["RIDAGEYR"] >= 18].reset_index(drop=True)
    is_female = df["RIAGENDR"] == 2

    # ── Canonical checkup-safe features (NaN preserved; each model imputes with
    # its OWN training medians, exactly as it would for a real partial checkup) ──
    feats = pd.DataFrame()
    feats["patient_id"] = "nhanes21_" + df["SEQN"].astype(int).astype(str)
    feats["age"] = df["RIDAGEYR"].astype(float)
    feats["sex"] = np.where(is_female, 0.0, 1.0)
    feats["bmi"] = df["BMXBMI"].astype(float)
    feats["systolic_bp"] = avg_bp(df, sy)
    feats["diastolic_bp"] = avg_bp(df, di)
    feats["glucose"] = df["LBXGLU"].astype(float)
    feats["cholesterol"] = df["LBXTC"].astype(float)
    feats["smoking"] = derive_smoking(df["SMQ020"], df["SMQ040"])
    feats["waist_circumference"] = df["BMXWAIST"].astype(float)
    feats["height"] = df["BMXHT"].astype(float)
    feats["resting_pulse"] = avg_bp(df, pls)
    feats["uric_acid"] = df["LBXSUA"].astype(float)
    feats["bun"] = df["LBXSBU"].astype(float)
    feats["triglycerides"] = df["LBXSTR"].astype(float)
    feats["cigs_per_day"] = pd.to_numeric(df["SMD650"], errors="coerce")
    feats.loc[feats["smoking"] == 0.0, "cigs_per_day"] = feats.loc[feats["smoking"] == 0.0, "cigs_per_day"].fillna(0.0)
    feats["heartRate"] = feats["resting_pulse"]
    feats["prevalentHyp"] = np.where(df["BPQ020"] == 1, 1.0, np.where(df["BPQ020"] == 2, 0.0, np.nan))

    # ── Independent ground-truth labels ──
    ghb = df["LBXGH"]
    diq = df["DIQ010"]
    dia_label = pd.Series(np.nan, index=df.index)
    dia_label[(ghb >= 6.5) | (diq == 1)] = 1
    dia_label[dia_label.isna() & (diq == 2) & (ghb.isna() | (ghb < 6.5))] = 0

    bpq = df["BPQ020"]
    ht_label = pd.Series(np.nan, index=df.index)
    ht_label[bpq == 1] = 1
    ht_label[bpq == 2] = 0

    c, d, e = df["MCQ160C"], df["MCQ160D"], df["MCQ160E"]
    heart_label = pd.Series(np.nan, index=df.index)
    heart_label[(c == 1) | (d == 1) | (e == 1)] = 1
    heart_label[(c == 2) & (d == 2) & (e == 2)] = 0

    egfr = egfr_ckd_epi_2021(df["LBXSCR"], df["RIDAGEYR"], is_female)
    ckd_lbl = pd.Series(ckd_label(egfr, df["URDACT"]), index=df.index).astype(float)
    ckd_lbl[df["LBXSCR"].isna()] = np.nan  # require creatinine to trust the label

    labels = pd.DataFrame({
        "diabetes": dia_label, "hypertension": ht_label,
        "heart": heart_label, "ckd": ckd_lbl,
    })
    return feats, labels


def evaluate(name, y_true, y_prob):
    mask = y_true.notna()
    y_t = y_true[mask].astype(int).to_numpy()
    y_p = y_prob[mask].to_numpy()
    if len(np.unique(y_t)) < 2:
        return None
    y_pred = (y_p >= 0.5).astype(int)
    auc = roc_auc_score(y_t, y_p)
    acc = accuracy_score(y_t, y_pred)
    f1 = f1_score(y_t, y_pred)
    conf = confusion_matrix(y_t, y_pred).ravel().tolist()  # TN, FP, FN, TP
    prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=10, strategy="quantile")
    return {
        "n": int(mask.sum()), "pos_rate": float(y_t.mean()),
        "auc": float(auc), "accuracy": float(acc), "f1": float(f1),
        "confusion": conf, "prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist(),
    }


def main():
    print("Building external NHANES 2021-2023 cohort...")
    feats, labels = build_cohort()
    print(f"  cohort size: {len(feats)} adults")
    for c in ["diabetes", "hypertension", "heart", "ckd"]:
        n = labels[c].notna().sum()
        pos = labels[c].mean(skipna=True)
        print(f"  {c:13s} labelled: {n:5d}  positive rate: {pos:.1%}")

    print("\nRunning chained predictions...")
    # Diabetes predictions are still needed as a chaining input for Heart,
    # but are NOT evaluated here — this cohort is now the Diabetes model's own
    # training data (see module docstring), so scoring it would be in-sample.
    # (Hypertension isn't chained into any other model, so it isn't computed
    # here at all — it's excluded from this table for the same in-sample
    # reason as Diabetes.)
    p_dia = diabetes_model.predict_risk_batch(feats)["diabetes_risk"]
    p_ckd = ckd_model.predict_risk_batch(feats)["ckd_risk"]
    feats_chain = feats.copy()
    feats_chain["diabetes_risk"] = p_dia.to_numpy()
    feats_chain["ckd_risk"] = p_ckd.to_numpy()
    p_heart = heart_model.predict_risk_batch(feats_chain)["heart_risk"]

    results = {}
    results["CKD"] = evaluate("CKD", labels["ckd"], p_ckd)
    results["Heart Disease"] = evaluate("Heart Disease", labels["heart"], p_heart)

    print("\n=== EXTERNAL VALIDATION RESULTS (NHANES 2021-2023, unseen cohort) ===")
    for name, r in results.items():
        if r is None:
            print(f"  {name}: insufficient labelled data")
            continue
        print(f"  {name:15s} n={r['n']:5d}  pos_rate={r['pos_rate']:.1%}  "
              f"AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}")

    joblib.dump(results, OUT_CACHE)
    write_report(results)
    print(f"\nSaved: {OUT_CACHE}")
    print(f"Saved: {OUT_MD}")


def write_report(results):
    with open(OUT_MD, "w") as f:
        f.write("# External Validation — NHANES August 2021-August 2023\n\n")
        f.write(
            "The CKD and Heart Disease shipped models were tested on a cohort "
            "**neither was trained on**: NHANES August 2021-August 2023, a "
            "nationally representative US survey of different people than either "
            "training dataset (CKD: NHANES 2017-2018; Heart: Framingham). "
            "Ground-truth labels are derived independently of our models, from "
            "objective labs and doctor-diagnosis questionnaire answers "
            "(doctor-diagnosed heart disease, eGFR + urine ACR for CKD) — never "
            "from our own predictions, so there is no circularity.\n\n"
            "**Diabetes and Hypertension are excluded from this table.** As of "
            "the P0 monotonic-constraints fix and the P1 hypertension dataset "
            "swap, both models now train directly on this same NHANES 2021-2023 "
            "cohort — Diabetes replacing the synthetic 100k Kaggle dataset that "
            "caused a fasting-glucose=158 mg/dL patient to be scored 4.1% "
            "\"LOW\" risk, Hypertension replacing the cardio/Kaggle dataset's "
            "broad self-reported cardiovascular-disease label with a real "
            "doctor-diagnosed-hypertension one. Scoring either model on this "
            "cohort would therefore be in-sample evaluation, not external "
            "validation — their honest out-of-sample estimates are the 5-fold "
            "CV numbers in `reports/baseline_metrics.md` (Diabetes: Accuracy "
            "81.19%, AUC 0.8624; Hypertension: Accuracy 73.20%, AUC 0.8077).\n\n"
        )
        f.write("## Results\n\n")
        f.write("| Disease | N (labelled) | Prevalence | AUC | Accuracy | F1 | Confusion (TN,FP,FN,TP) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, r in results.items():
            if r is None:
                f.write(f"| {name} | — | — | insufficient data | — | — | — |\n")
                continue
            f.write(f"| {name} | {r['n']} | {r['pos_rate']:.1%} | {r['auc']:.4f} | "
                    f"{r['accuracy']:.2%} | {r['f1']:.4f} | {r['confusion']} |\n")

        f.write("\n## How to read this\n\n")
        f.write(
            "- This is a **stricter** test than the 5-fold CV or the 80/20 held-out "
            "split reported in `reports/baseline_metrics.md` — those come from the "
            "SAME dataset each model was trained on. This cohort is entirely new "
            "people, from a different survey wave.\n"
            "- A model that scores similarly here to its training-time metrics has "
            "**genuinely generalized**. A large drop indicates the training metric "
            "was optimistic (overfit to that dataset's quirks) or the label "
            "definitions do not perfectly match across datasets (see caveats).\n"
        )
        f.write("\n## Caveats\n\n")
        f.write(
            "- **Label definitions are not identical to each training target.** "
            "Heart Disease compares against doctor-diagnosed CHD/angina/"
            "heart-attack, the closest analogue to Framingham's 10-year CHD "
            "outcome (which is prospective, not point-in-time).\n"
            "- Some features are missing per person (e.g. fasting glucose is only "
            "collected on a NHANES sub-sample); each model fills gaps with its own "
            "training medians, exactly as it would for a real partial checkup.\n"
            "- CKD requires serum creatinine to be present (for eGFR); rows without "
            "it are excluded from CKD evaluation only.\n"
        )


if __name__ == "__main__":
    main()
