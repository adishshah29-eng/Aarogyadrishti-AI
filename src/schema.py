"""
Canonical feature schema for AarogyaDrishti AI.

Defines the shared feature set used across all datasets and models, together
with the **units** each feature is expressed in and the **plausible clinical
range** for validation. Keeping units here is what prevents the class of bug
where one dataset encodes a feature differently from another (e.g. cholesterol
as mg/dL vs an ordinal 1-3 category): every processed dataset is cleaned to
these canonical units, and the dashboard validates inputs against these ranges.
"""

# The canonical list of features collected at checkup time.
CANONICAL_FEATURES = [
    'patient_id',
    'age',
    'sex',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'glucose',
    'cholesterol',
    'smoking',
    'alcohol',
    'physical_activity',
    'family_history',
]

# Canonical unit for each numeric feature. All processed datasets and all model
# inputs must use these units.
FEATURE_UNITS = {
    'age':               'years',
    'sex':               'binary (1=male, 0=female)',
    'bmi':               'kg/m^2',
    'systolic_bp':       'mmHg',
    'diastolic_bp':      'mmHg',
    'glucose':           'mg/dL',
    'cholesterol':       'mg/dL',
    'smoking':           'binary (1=smoker)',
    'alcohol':           'binary (1=yes)',
    'physical_activity': 'binary (1=active)',
    'family_history':    'reserved (not populated by any shipped model)',
}

# Physiologically plausible ranges, used by the dashboard to warn on bad input.
FEATURE_RANGES = {
    'age':         (1, 110),
    'bmi':         (13.0, 60.0),
    'systolic_bp': (70, 220),
    'diastolic_bp':(40, 140),
    'glucose':     (50, 600),
    'cholesterol': (100, 500),
}


def range_warnings(values: dict) -> list:
    """Return a list of (field, message) warnings for any value outside its
    canonical plausible range. `values` maps feature name -> numeric value.
    Also enforces the systolic > diastolic relationship when both are present.
    """
    labels = {
        'age': 'Age', 'bmi': 'BMI', 'systolic_bp': 'Systolic BP',
        'diastolic_bp': 'Diastolic BP', 'glucose': 'Fasting Glucose',
        'cholesterol': 'Cholesterol',
    }
    warnings = []
    for feat, (lo, hi) in FEATURE_RANGES.items():
        if feat in values and values[feat] is not None:
            v = values[feat]
            if not (lo <= v <= hi):
                unit = FEATURE_UNITS.get(feat, '')
                warnings.append((labels[feat], f"{v} is outside the expected range {lo}–{hi} {unit}."))
    sbp, dbp = values.get('systolic_bp'), values.get('diastolic_bp')
    if sbp is not None and dbp is not None and sbp <= dbp:
        warnings.append(("BP relationship", f"Systolic BP ({sbp}) must be greater than Diastolic BP ({dbp})."))
    return warnings
