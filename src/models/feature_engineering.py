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
