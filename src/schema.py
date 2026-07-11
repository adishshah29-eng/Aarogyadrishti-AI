"""
Canonical feature schema for AarogyaDrishti AI.
Defines the shared feature set across all models and datasets.
"""

# The canonical list of features collected at checkup time
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
    'family_history'
]
