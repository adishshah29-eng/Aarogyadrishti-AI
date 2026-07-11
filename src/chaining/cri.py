import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.diabetes_model import predict_risk as pred_dia
from src.models.ckd_model import predict_risk as pred_ckd
from src.models.heart_model import predict_risk as pred_heart
from src.models.hypertension_model import predict_risk as pred_ht

def compute_cri(diabetes_risk: float, ckd_risk: float, heart_risk: float, hypertension_risk: float) -> float:
    """
    Compute the Comorbidity Risk Index (CRI) for a patient.
    Formula:
      CRI = 0.30 * P(Diabetes) + 0.20 * P(CKD) + 0.25 * P(Heart Disease) + 0.25 * P(Hypertension) + Interactions
      
    Interactions:
      - Diabetes + Heart: +0.15 * P(Diabetes) * P(Heart Disease) (diabetics are 2-4x more likely to develop HD)
      - Hypertension + Heart: +0.10 * P(Hypertension) * P(Heart Disease)
      - Diabetes + Hypertension: +0.05 * P(Diabetes) * P(Hypertension)
      
    The final score is clipped to [0.0, 1.0].
    """
    # Base weighted sum
    base_score = (
        0.30 * diabetes_risk +
        0.20 * ckd_risk +
        0.25 * heart_risk +
        0.25 * hypertension_risk
    )
    
    # Interaction terms
    interactions = (
        0.15 * diabetes_risk * heart_risk +
        0.10 * hypertension_risk * heart_risk +
        0.05 * diabetes_risk * hypertension_risk
    )
    
    cri = base_score + interactions
    return float(np.clip(cri, 0.0, 1.0))

def get_full_risk_profile(patient_df) -> dict:
    """
    Exposes full risk profile handoff interface.
    Chains predictions:
      1. Predict diabetes and CKD risk.
      2. Append these risks to patient details.
      3. Predict hypertension and heart disease risk.
      4. Calculate CRI.
      5. Return dict of all risk scores.
    """
    # Parse input into DataFrame
    if isinstance(patient_df, dict):
        df = pd.DataFrame([patient_df])
    elif isinstance(patient_df, pd.Series):
        df = pd.DataFrame([patient_df])
    else:
        df = patient_df.copy()
        
    # Step 1: Predict upstream risks
    p_dia = pred_dia(df)
    p_ckd = pred_ckd(df)
    
    # Step 2: Append predicted risks as features for chained predictors
    df['diabetes_risk'] = p_dia
    df['ckd_risk'] = p_ckd
    
    # Step 3: Predict downstream risks (using the chained models)
    p_ht = pred_ht(df)
    p_heart = pred_heart(df)
    
    # Step 4: Calculate comorbidity index
    cri = compute_cri(p_dia, p_ckd, p_heart, p_ht)
    
    return {
        'diabetes_risk': p_dia,
        'ckd_risk': p_ckd,
        'hypertension_risk': p_ht,
        'heart_risk': p_heart,
        'cri': cri
    }

def run_cri_validation():
    print("=== CRI Weighting Rationale Validation ===")
    
    # Validate against 3 hand-checked patient profiles
    test_cases = [
        ("Patient A (All Low Risks)", 0.1, 0.05, 0.1, 0.1),
        ("Patient B (Single High Risk - Heart)", 0.1, 0.05, 0.8, 0.1),
        ("Patient C (Comorbidity High Risks - Diabetes + Heart)", 0.7, 0.1, 0.8, 0.1)
    ]
    
    for name, dia, ckd, heart, ht in test_cases:
        cri = compute_cri(dia, ckd, heart, ht)
        print(f"\n{name}:")
        print(f"  Inputs -> Diabetes: {dia:.2f}, CKD: {ckd:.2f}, Heart: {heart:.2f}, HT: {ht:.2f}")
        print(f"  Calculated CRI: {cri:.4f}")
        
        # Verify clinical properties
        if "Low" in name:
            assert cri < 0.20, f"Low risk patient has too high CRI: {cri}"
        elif "Single" in name:
            # 0.3*0.1 + 0.2*0.05 + 0.25*0.8 + 0.25*0.1 + interactions
            # = 0.03 + 0.01 + 0.20 + 0.025 + (0.15*0.08 + 0.1*0.08 + 0.05*0.01)
            # = 0.265 + (0.012 + 0.008 + 0.0005) = 0.265 + 0.0205 = 0.2855
            assert abs(cri - 0.2855) < 1e-4, f"CRI calculation mismatch for Patient B: {cri}"
        elif "Comorbidity" in name:
            # 0.3*0.7 + 0.2*0.1 + 0.25*0.8 + 0.25*0.1 = 0.21 + 0.02 + 0.20 + 0.025 = 0.455
            # Interactions:
            # - dia*heart: 0.15 * 0.7 * 0.8 = 0.084
            # - ht*heart: 0.10 * 0.1 * 0.8 = 0.008
            # - dia*ht: 0.05 * 0.7 * 0.1 = 0.0035
            # Total: 0.455 + 0.0955 = 0.5505
            assert abs(cri - 0.5505) < 1e-4, f"CRI calculation mismatch for Patient C: {cri}"
            print("  Clinical amplification verify: CRI was pushed up from base sum 0.455 to 0.5505 (+21.0% relative increase) due to diabetes + heart disease comorbidity. [OK]")
            
    print("\nCRI logic validation PASSED! [OK]")

if __name__ == "__main__":
    run_cri_validation()
