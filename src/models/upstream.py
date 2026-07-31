"""
Shared helper for the chaining architecture.

Heart Disease and Hypertension are *downstream* models: they consume the
predicted Diabetes and CKD risk of the same patient as two extra input
features. This module is the single place that turns a canonical-schema
patient frame into that chained feature frame, so training, evaluation, and
inference all build the upstream features identically.

Note on cross-dataset imputation: the per-disease training CSVs do not all
carry the columns the upstream models need (e.g. the heart dataset has no
glucose/BMI), so those columns are median-imputed by the upstream predictors
at train time. At dashboard inference the full checkup panel is present, so
the upstream risks there carry more signal than they did during training.
This distribution shift is an inherent limitation of chaining models that
were each trained on a different public dataset — see reports/chaining_results.md.
"""
import pandas as pd

UPSTREAM_FEATURES = ["diabetes_risk", "ckd_risk"]


def add_upstream_risks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``diabetes_risk`` and ``ckd_risk`` columns
    appended, computed by the trained upstream (Diabetes, CKD) models.

    Requires ``models/diabetes_model.pkl`` and ``models/ckd_model.pkl`` to
    already exist (train the upstream models before the downstream ones).
    Imports are local to avoid a circular import at module load time.
    """
    from src.models.diabetes_model import predict_risk_batch as pred_dia_batch
    from src.models.ckd_model import predict_risk_batch as pred_ckd_batch

    df = df.copy()
    df["diabetes_risk"] = pred_dia_batch(df)["diabetes_risk"].to_numpy()
    df["ckd_risk"] = pred_ckd_batch(df)["ckd_risk"].to_numpy()
    return df
