"""
Registry of every value a lab report can fill in.

Each entry ties a wizard field (its PDATA key) to: the canonical unit the
models were trained on, the exact bounds of the wizard's number_input (a
pre-filled value outside them makes Streamlit raise), the units a real report
may print it in, and the model features it ultimately feeds. Which models a
field affects is read from each model's own feature list, so this file can't
drift from what the models actually consume.
"""
from dataclasses import dataclass
from typing import Optional

from src.models import diabetes_model, ckd_model, heart_model, hypertension_model


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    step: int                     # wizard step that collects it; 0 = context only
    required: bool                # wizard has no "I don't know" for it
    unit: str                     # canonical unit
    lo: float                     # wizard widget bounds (hard accept limits)
    hi: float
    integer: bool                 # wizard widget takes ints
    model_inputs: tuple           # model feature names this value feeds
    aliases: tuple                # names labs print it under (prompt hints)
    units: tuple = ()              # (normalized unit, multiplier to canonical)
    idk_key: Optional[str] = None  # wizard "I don't know" checkbox key
    categorical: bool = False


FIELDS = (
    FieldSpec("age", "Age", 1, True, "years", 1, 120, True, ("age",),
              ("Age", "Age/Sex", "Age/Gender"),
              (("years", 1.0), ("yrs", 1.0), ("yr", 1.0), ("y", 1.0))),
    FieldSpec("sex", "Sex", 1, True, "", 0, 0, False, ("sex",),
              ("Sex", "Gender", "Age/Sex"), categorical=True),
    FieldSpec("height", "Height", 1, True, "cm", 50.0, 250.0, False, ("bmi",),
              ("Height", "Ht"),
              (("cm", 1.0), ("m", 100.0), ("in", 2.54), ("inch", 2.54), ("inches", 2.54))),
    FieldSpec("weight", "Weight", 1, True, "kg", 10.0, 300.0, False, ("bmi",),
              ("Weight", "Wt", "Body weight"),
              (("kg", 1.0), ("kgs", 1.0), ("lb", 0.45359237), ("lbs", 0.45359237))),

    FieldSpec("systolic_bp", "Systolic BP", 2, True, "mmHg", 70, 250, True, ("systolic_bp",),
              ("Blood pressure (upper number)", "SBP", "BP"), (("mmhg", 1.0),)),
    FieldSpec("diastolic_bp", "Diastolic BP", 2, True, "mmHg", 40, 150, True, ("diastolic_bp",),
              ("Blood pressure (lower number)", "DBP", "BP"), (("mmhg", 1.0),)),
    FieldSpec("glucose", "Fasting glucose", 2, True, "mg/dL", 50, 400, True, ("glucose",),
              ("Glucose Fasting", "Fasting Blood Sugar", "FBS", "FBG", "Plasma Glucose (F)",
               "Blood Sugar Fasting"),
              (("mg/dl", 1.0), ("mmol/l", 18.016))),
    FieldSpec("cholesterol", "Total cholesterol", 2, True, "mg/dL", 100, 400, True, ("cholesterol",),
              ("Total Cholesterol", "Cholesterol, Total", "S. Cholesterol", "Serum Cholesterol"),
              (("mg/dl", 1.0), ("mmol/l", 38.67))),

    FieldSpec("waist_circumference", "Waist circumference", 2, False, "cm", 40.0, 200.0, False,
              ("waist_circumference",), ("Waist circumference", "Waist"),
              (("cm", 1.0), ("in", 2.54), ("inch", 2.54), ("inches", 2.54)), idk_key="idk_waist"),
    FieldSpec("resting_pulse", "Resting pulse", 2, False, "bpm", 30, 200, True,
              ("resting_pulse", "heartRate"), ("Pulse", "Pulse rate", "Heart rate", "HR"),
              (("bpm", 1.0), ("/min", 1.0), ("beats/min", 1.0), ("per min", 1.0)), idk_key="idk_pulse"),
    FieldSpec("uric_acid", "Uric acid", 2, False, "mg/dL", 1.0, 15.0, False, ("uric_acid",),
              ("Uric Acid", "S. Uric Acid", "Serum Uric Acid"),
              (("mg/dl", 1.0), ("umol/l", 1 / 59.48), ("mmol/l", 16.81)), idk_key="idk_uric"),
    # BUN is special-cased in normalize.py: Indian labs usually print *urea*,
    # which must be converted to urea *nitrogen* before it means BUN.
    FieldSpec("bun", "Blood urea nitrogen (BUN)", 2, False, "mg/dL", 3.0, 90.0, False, ("bun",),
              ("BUN", "Blood Urea Nitrogen", "Urea Nitrogen", "Blood Urea", "Urea", "Serum Urea"),
              (("mg/dl", 1.0), ("mmol/l", 2.801)), idk_key="idk_bun"),
    FieldSpec("triglycerides", "Triglycerides", 2, False, "mg/dL", 20.0, 1000.0, False,
              ("triglycerides",), ("Triglycerides", "S. Triglycerides", "TG"),
              (("mg/dl", 1.0), ("mmol/l", 88.57)), idk_key="idk_trig"),

    FieldSpec("hba1c", "HbA1c", 0, False, "%", 3.0, 20.0, False, (),
              ("HbA1c", "Glycated Haemoglobin", "Glycosylated Hemoglobin", "A1c"),
              (("%", 1.0),)),
)

SPECS = {f.key: f for f in FIELDS}

# What only the patient can tell us — never on a lab report.
SELF_REPORTED = (
    "Smoking status (and cigarettes per day)",
    "Ever diagnosed with high blood pressure",
    "Currently on blood pressure medication",
)

MODEL_RAW_FEATURES = {
    "Diabetes": tuple(diabetes_model.RAW_FEATURES),
    "CKD": tuple(ckd_model.RAW_CHECKUP_FEATURES),
    "Heart Disease": tuple(heart_model.RAW_ISOLATED_FEATURES),
    "Hypertension": tuple(hypertension_model.RAW_ISOLATED_FEATURES),
}


def models_using(spec: FieldSpec) -> list:
    return [name for name, feats in MODEL_RAW_FEATURES.items()
            if any(m in feats for m in spec.model_inputs)]
