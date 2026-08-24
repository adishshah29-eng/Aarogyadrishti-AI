"""
Shared feature-engineering transforms used across the four risk models.

Every function takes a DataFrame that already has the required raw columns
(imputed with training medians if necessary) and returns it with one new
column appended. Applying these at both training and prediction time keeps
the derived features numerically identical in both paths.
"""
import pandas as pd


def add_pulse_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Pulse pressure = systolic - diastolic BP. Wide pulse pressure is an
    independent marker of arterial stiffness and cardiovascular risk."""
    df = df.copy()
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    return df


def add_mean_arterial_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """MAP = diastolic + 1/3 * pulse pressure. Approximates average perfusion
    pressure over the cardiac cycle."""
    df = df.copy()
    pulse_pressure = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + pulse_pressure / 3.0
    return df


def add_waist_height_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Waist-to-height ratio, a central-adiposity marker that outperforms BMI
    alone for metabolic risk in several cohort studies."""
    df = df.copy()
    df["waist_height_ratio"] = df["waist_circumference"] / df["height"]
    return df


def add_age_glucose_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Age x glucose (scaled). Diabetes risk from a given glucose level rises
    faster with age; this lets the tree split on that combined effect
    directly instead of approximating it with separate age/glucose splits."""
    df = df.copy()
    df["age_glucose_interaction"] = (df["age"] * df["glucose"]) / 1000.0
    return df


def add_bmi_age_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """BMI x age (scaled). Same rationale as age_glucose_interaction, applied
    to the BMI/age relationship relevant to CKD risk."""
    df = df.copy()
    df["bmi_age_interaction"] = (df["bmi"] * df["age"]) / 100.0
    return df


def add_lipid_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Cholesterol x triglycerides (scaled). The single most informative
    non-redundant feature-pair interaction found by SHAP interaction-value
    analysis of the trained diabetes model (scripts/analyze_shap_interactions.py)
    — an elevated-lipid-profile signal that neither value captures alone."""
    df = df.copy()
    df["lipid_interaction"] = (df["cholesterol"] * df["triglycerides"]) / 10000.0
    return df


def add_uric_acid_sex_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Uric acid expressed as a ratio to its sex-specific clinical upper
    limit (7.2 mg/dL men, 6.0 mg/dL women), so the same raw value carries
    different meaning by sex the way it clinically should. SHAP interaction
    analysis of the trained CKD model found sex x uric_acid to be its
    second-strongest feature-pair interaction, after sex's own broad
    interaction with the age x BMI term."""
    df = df.copy()
    upper_limit = df["sex"].map({1.0: 7.2, 0.0: 6.0}).fillna(6.6)
    df["uric_acid_sex_norm"] = df["uric_acid"] / upper_limit
    return df


def add_comorbidity_risk_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """diabetes_risk x ckd_risk. The single strongest feature-pair interaction
    SHAP found in the trained Hypertension model — comorbid diabetes and CKD
    risk compound rather than simply add, echoing the same amplification
    already assumed by the CRI formula's interaction terms (src/chaining/cri.py)
    but, until now, never exposed to the chained models themselves."""
    df = df.copy()
    df["comorbidity_risk_interaction"] = df["diabetes_risk"] * df["ckd_risk"]
    return df
