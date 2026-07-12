import os
import yaml
import pandas as pd

class InsightGenerator:
    """
    A generic explanation generator that interprets SHAP values using external clinical guidelines.
    Adheres to ML4H separation of concerns by keeping clinical knowledge out of the UI.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to the adjacent config folder
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "clinical_guidelines.yaml")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.features = self.config.get("features", {})
        
    def _format_value(self, feat: str, raw_val, unit: str) -> str:
        if raw_val is None:
            return ""
            
        if isinstance(raw_val, float):
            if "risk" in feat.lower() and raw_val <= 1.0:
                # Format 0-1 risk scores as percentages
                return f" ({raw_val*100:.1f}{unit})"
            else:
                return f" ({raw_val:.1f}{unit})"
        return f" ({raw_val}{unit})"

    def generate(self, shap_dict: dict, patient_data: dict = None, top_k: int = 4) -> str:
        """
        Generate detailed HTML insights from SHAP values.
        Supports both risk-elevating (positive SHAP) and protective (negative SHAP) features.
        """
        if not shap_dict:
            return "No clear risk drivers identified."
            
        # Sort features by absolute impact magnitude
        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Split into elevating (positive) and protective (negative) drivers
        elevating = [(feat, val) for feat, val in sorted_features if val > 0][:top_k]
        protective = [(feat, val) for feat, val in sorted_features if val < 0][:top_k]
        
        lines = []
        
        # Process Elevating Factors (Needs Improvement)
        if elevating:
            lines.append("<div style='margin-bottom: 12px;'>")
            lines.append("<strong>Factors elevating your risk:</strong>")
            lines.append("<ul style='margin-top: 4px; margin-bottom: 0px;'>")
            for feat, _ in elevating:
                feat_cfg = self.features.get(feat, {})
                friendly_name = feat_cfg.get("name", feat.replace('_', ' ').title())
                unit = feat_cfg.get("unit", "")
                advice = feat_cfg.get("positive_advice", "This factor is contributing to your overall risk profile.")
                
                raw_val = patient_data.get(feat) if patient_data else None
                val_str = self._format_value(feat, raw_val, unit)
                
                lines.append(f"<li style='margin-bottom: 4px;'><strong>{friendly_name}{val_str}</strong>: {advice}</li>")
            lines.append("</ul></div>")
        else:
            lines.append("<p>Your risk is currently very low, with no major factors driving it up.</p>")
            
        # Process Protective Factors (Doing Well)
        if protective:
            lines.append("<div style='margin-top: 12px;'>")
            lines.append("<strong style='color: var(--teal-600);'>Factors protecting your health:</strong>")
            lines.append("<ul style='margin-top: 4px; margin-bottom: 0px;'>")
            for feat, _ in protective:
                feat_cfg = self.features.get(feat, {})
                friendly_name = feat_cfg.get("name", feat.replace('_', ' ').title())
                unit = feat_cfg.get("unit", "")
                advice = feat_cfg.get("negative_advice", "This factor is actively lowering your risk.")
                
                raw_val = patient_data.get(feat) if patient_data else None
                val_str = self._format_value(feat, raw_val, unit)
                
                lines.append(f"<li style='margin-bottom: 4px;'><strong>{friendly_name}{val_str}</strong>: {advice}</li>")
            lines.append("</ul></div>")
            
        return "".join(lines)
