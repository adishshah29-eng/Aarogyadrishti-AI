"""
Plain-language explanation templates for SHAP values.
"""

MODIFIABLE_FEATURES = {
    "bmi": ("BMI", ""),
    "systolic_bp": ("systolic blood pressure", " mmHg"),
    "diastolic_bp": ("diastolic blood pressure", " mmHg"),
    "glucose": ("fasting glucose", " mg/dL"),
    "cholesterol": ("total cholesterol", " mg/dL"),
    "smoking": ("smoking habit", ""),
    "alcohol": ("alcohol intake", ""),
    "physical_activity": ("physical activity level", "")
}

def get_personalized_insights(shap_dict: dict, patient_data: dict = None, top_k: int = 3) -> str:
    """
    Convert a dictionary of SHAP values into a highly personalized plain-language explanation.
    """
    if not shap_dict:
        return "No clear risk drivers identified."
        
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_drivers = sorted_features[:top_k]
    
    # Filter for modifiable risk factors that positively contribute to the risk
    modifiable_drivers = [feat for feat, val in top_drivers if feat in MODIFIABLE_FEATURES and val > 0]
    
    if not modifiable_drivers:
        return "Your primary risk factors are mostly non-modifiable (e.g., age, family history). Continue regular checkups to monitor your health."
        
    statements = []
    for feat in modifiable_drivers:
        friendly_name, unit = MODIFIABLE_FEATURES[feat]
        val_str = ""
        if patient_data and feat in patient_data:
            val = patient_data[feat]
            # Format number cleanly if it's a float
            if isinstance(val, float):
                val_str = f" of {val:.1f}{unit}"
            else:
                val_str = f" of {val}{unit}"
                
        statements.append(f"**{friendly_name}**{val_str}")
        
    drivers_str = " and ".join(statements)
    
    template = f"We noticed that your {drivers_str} strongly elevated your risk score here. "
    template += "Focusing on these specific lifestyle areas with your healthcare provider could yield the biggest improvements for your long-term health."
    
    return template
