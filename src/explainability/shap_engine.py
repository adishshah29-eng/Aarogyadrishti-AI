"""
SHAP Engine for AarogyaDrishti AI
"""
import shap
import pandas as pd

def explain_prediction(model, patient_row: pd.DataFrame) -> dict:
    """
    Generate SHAP values for a given patient prediction.
    
    Args:
        model: A tree-based model (e.g., XGBoost, RandomForest)
        patient_row (pd.DataFrame): A single-row DataFrame containing the patient's features.
        
    Returns:
        dict: A dictionary mapping feature names to their corresponding SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_row)
    
    if isinstance(shap_values, list):
        values = shap_values[1][0] 
    else:
        values = shap_values[0]
        
    feature_names = patient_row.columns.tolist()
    shap_dict = {feat: float(val) for feat, val in zip(feature_names, values)}
    
    return shap_dict

def get_shap_explanation(model, patient_row: pd.DataFrame):
    """
    Returns the raw shap.Explanation object for use with shap.plots (e.g. waterfall).
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(patient_row)
    if len(shap_values.shape) == 3: # For multiclass
        return shap_values[0, :, 1]
    return shap_values[0]
