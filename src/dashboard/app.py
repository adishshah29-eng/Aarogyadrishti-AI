import os
import sys
import joblib
import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chaining.cri import get_full_risk_profile
from src.explainability.shap_engine import explain_prediction
from src.schema import range_warnings

from src.models.diabetes_model import (_load_model_data as load_diabetes,
    _engineer as engineer_diabetes, RAW_FEATURES as RAW_DIABETES_FEATURES)
from src.models.ckd_model import (_load_model_data as load_ckd,
    _engineer as engineer_ckd, RAW_CHECKUP_FEATURES as RAW_CKD_FEATURES)
from src.models.heart_model import (_load_model_data as load_heart,
    _engineer as engineer_heart, RAW_ISOLATED_FEATURES as RAW_HEART_FEATURES)
from src.models.hypertension_model import (_load_model_data as load_hypertension,
    _engineer as engineer_hypertension, RAW_ISOLATED_FEATURES as RAW_HT_FEATURES)

st.set_page_config(
    page_title="AarogyaDrishti AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System: Clinical Light ──────────────────────────────────────────────
# Palette: Warm White + Cool Slate + Teal Primary + Coral/Amber status
# Inspired by: Linear, Craft, modern EMR systems
# Fonts: Plus Jakarta Sans (sharp, modern medical) + Inter (body)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
  /* Surfaces */
  --bg:            #F7F8FA;
  --surface:       #FFFFFF;
  --surface-raised:#FFFFFF;
  --surface-muted: #F0F2F5;
  --surface-hover: #EEF0F3;

  /* Borders */
  --border:        #E2E5EA;
  --border-strong: #C8CDD6;

  /* Brand: Teal — clinical, trustworthy, modern */
  --teal-50:  #EDFAFA;
  --teal-100: #D5F5F6;
  --teal-500: #0891B2;
  --teal-600: #0771A0;
  --teal-700: #065A87;

  /* Status */
  --low:       #0D9488;   /* teal-600 — calm */
  --low-bg:    #F0FDFB;
  --low-border:#99F6E4;
  --mid:       #D97706;   /* warm amber */
  --mid-bg:    #FFFBEB;
  --mid-border:#FCD34D;
  --high:      #DC2626;   /* no-nonsense red */
  --high-bg:   #FEF2F2;
  --high-border:#FECACA;

  /* Text */
  --text-primary:   #0F1923;
  --text-secondary: #4A5568;
  --text-muted:     #8896A7;
  --text-on-teal:   #FFFFFF;

  /* Typography */
  --font-head: 'Plus Jakarta Sans', sans-serif;
  --font-body: 'Inter', sans-serif;

  /* Elevation */
  --shadow-xs: 0 1px 2px rgba(15,25,35,0.06);
  --shadow-sm: 0 2px 8px rgba(15,25,35,0.08);
  --shadow-md: 0 4px 20px rgba(15,25,35,0.10);
  --shadow-lg: 0 8px 40px rgba(15,25,35,0.12);

  /* Radii */
  --r-sm: 6px;
  --r-md: 10px;
  --r-lg: 16px;
  --r-xl: 20px;

  /* Transition */
  --ease: all 0.2s cubic-bezier(0.4,0,0.2,1);
}

/* ── Global ── */
html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}
.stApp {
  background-color: var(--bg) !important;
}
.stApp > header { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

/* ── Main container ── */
.main .block-container {
  padding: 0 2rem 2rem !important;
  max-width: 1380px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Sidebar brand strip */
.sb-brand {
  background: linear-gradient(135deg, #0891B2 0%, #0D9488 100%);
  padding: 20px 18px 16px;
  margin-bottom: 4px;
}
.sb-brand-name {
  font-family: var(--font-head) !important;
  font-size: 1rem !important;
  font-weight: 800 !important;
  color: #fff !important;
  letter-spacing: -0.01em;
  margin: 0 !important;
}
.sb-brand-sub {
  font-size: 0.72rem !important;
  color: rgba(255,255,255,0.75) !important;
  margin: 2px 0 0 !important;
  letter-spacing: 0.02em;
}

/* Sidebar section headers */
.sb-section {
  font-family: var(--font-head) !important;
  font-size: 0.65rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--teal-500) !important;
  padding: 14px 0 4px !important;
  margin: 0 !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] label {
  font-family: var(--font-body) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--surface-muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font-body) !important;
  font-size: 0.84rem !important;
  color: var(--text-primary) !important;
  box-shadow: var(--shadow-xs) !important;
  transition: var(--ease) !important;
}
[data-testid="stSidebar"] .stNumberInput input:focus {
  border-color: var(--teal-500) !important;
  background: var(--surface) !important;
  box-shadow: 0 0 0 3px rgba(8,145,178,0.12) !important;
  outline: none !important;
}

/* Submit button */
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, #0891B2, #0D9488) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.01em !important;
  padding: 11px 20px !important;
  width: 100% !important;
  cursor: pointer !important;
  box-shadow: 0 3px 12px rgba(8,145,178,0.35) !important;
  transition: var(--ease) !important;
}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
  filter: brightness(1.06) !important;
  box-shadow: 0 6px 20px rgba(8,145,178,0.40) !important;
  transform: translateY(-1px) !important;
}

/* ── Top header bar ── */
.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 22px 0 18px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 28px;
}
.top-bar-left { display: flex; align-items: center; gap: 14px; }
.top-bar-icon {
  width: 44px; height: 44px;
  background: linear-gradient(135deg, #0891B2, #0D9488);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  box-shadow: 0 4px 14px rgba(8,145,178,0.30);
  flex-shrink: 0;
}
.top-bar-title {
  font-family: var(--font-head) !important;
  font-size: 1.45rem !important;
  font-weight: 800 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.025em !important;
  margin: 0 !important;
  line-height: 1.2 !important;
}
.top-bar-sub {
  font-size: 0.8rem !important;
  color: var(--text-muted) !important;
  margin: 2px 0 0 !important;
  font-weight: 400 !important;
}
.top-bar-badge {
  background: var(--teal-50);
  border: 1px solid var(--teal-100);
  color: var(--teal-600);
  font-family: var(--font-head);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 5px 10px;
  border-radius: 999px;
}

/* ── CRI Section ── */
.cri-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  box-shadow: var(--shadow-sm);
  padding: 28px 32px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 32px;
  position: relative;
  overflow: hidden;
}
.cri-wrap::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--r-xl) var(--r-xl) 0 0;
}
.cri-label {
  font-family: var(--font-head) !important;
  font-size: 0.68rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  margin: 0 0 4px !important;
}
.cri-number {
  font-family: var(--font-head) !important;
  font-size: 3.8rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.04em !important;
  line-height: 1 !important;
  margin: 0 !important;
}
.cri-desc {
  font-family: var(--font-body) !important;
  font-size: 0.85rem !important;
  color: var(--text-secondary) !important;
  line-height: 1.55 !important;
  margin: 10px 0 0 !important;
  max-width: 400px;
}

/* ── Status badge ── */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 999px;
  font-family: var(--font-head);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid;
}
.status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  display: inline-block;
}

/* ── Disease risk cards ── */
.dc-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 28px; }
.dc-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 20px 18px 16px;
  box-shadow: var(--shadow-xs);
  transition: var(--ease);
  position: relative;
  overflow: hidden;
}
.dc-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
  border-color: var(--border-strong);
}
.dc-card-accent {
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--r-lg) var(--r-lg) 0 0;
}
.dc-card-name {
  font-family: var(--font-head) !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  margin: 0 0 12px !important;
}
.dc-card-value {
  font-family: var(--font-head) !important;
  font-size: 2.4rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.04em !important;
  line-height: 1 !important;
  margin: 0 0 12px !important;
}
.dc-card-track {
  height: 5px;
  background: var(--surface-muted);
  border-radius: 3px;
  margin-bottom: 10px;
  overflow: hidden;
}
.dc-card-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 1s cubic-bezier(0.4,0,0.2,1);
}
.dc-card-note {
  font-family: var(--font-body) !important;
  font-size: 0.68rem !important;
  color: var(--text-muted) !important;
  line-height: 1.4 !important;
  margin: 0 !important;
}

/* ── Section header ── */
.sec-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 16px;
}
.sec-head-title {
  font-family: var(--font-head) !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.01em !important;
  margin: 0 !important;
}
.sec-head-sub {
  font-size: 0.78rem !important;
  color: var(--text-muted) !important;
  margin: 0 !important;
}

/* ── Insight card ── */
.insight-card {
  background: #FAFCFF;
  border: 1px solid #DDE8F5;
  border-left: 3px solid var(--teal-500);
  border-radius: var(--r-md);
  padding: 13px 16px;
  margin-bottom: 14px;
}
.insight-card p {
  font-family: var(--font-body) !important;
  font-size: 0.85rem !important;
  color: var(--text-secondary) !important;
  line-height: 1.6 !important;
  margin: 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface-muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 3px !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: var(--r-sm) !important;
  color: var(--text-muted) !important;
  font-family: var(--font-head) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  padding: 7px 14px !important;
  transition: var(--ease) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--surface) !important;
  color: var(--teal-600) !important;
  box-shadow: var(--shadow-xs) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--r-md) var(--r-md) !important;
  padding: 20px 16px !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: var(--r-md) !important; box-shadow: var(--shadow-xs) !important; }
[data-testid="stDataFrame"] th {
  background: var(--surface-muted) !important;
  font-family: var(--font-head) !important;
  font-weight: 600 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
  color: var(--text-secondary) !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 28px 0 !important;
}

/* ── Empty state ── */
.empty-wrap {
  text-align: center;
  padding: 56px 0 32px;
}
.empty-icon {
  width: 72px; height: 72px;
  background: linear-gradient(135deg, var(--teal-50), #EEF8FF);
  border: 1px solid var(--teal-100);
  border-radius: 20px;
  display: flex; align-items: center; justify-content: center;
  font-size: 32px;
  margin: 0 auto 20px;
}
.empty-wrap h3 {
  font-family: var(--font-head) !important;
  font-size: 1.25rem !important;
  font-weight: 800 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 8px !important;
}
.empty-wrap p {
  font-size: 0.88rem !important;
  color: var(--text-secondary) !important;
  line-height: 1.6 !important;
  margin: 0 !important;
}

/* ── How it works ── */
.hiw-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 32px; }
.hiw-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 18px 16px;
  box-shadow: var(--shadow-xs);
}
.hiw-num {
  width: 26px; height: 26px;
  background: linear-gradient(135deg, #0891B2, #0D9488);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-head);
  font-size: 0.72rem; font-weight: 800; color: #fff;
  margin-bottom: 10px;
}
.hiw-card h4 {
  font-family: var(--font-head) !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  margin: 0 0 4px !important;
  letter-spacing: -0.01em !important;
}
.hiw-card p {
  font-family: var(--font-body) !important;
  font-size: 0.74rem !important;
  color: var(--text-muted) !important;
  line-height: 1.5 !important;
  margin: 0 !important;
}

/* ── Wizard ── */
.wiz-steps {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 4px 0 28px;
}
.wiz-step {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 10px;
}
.wiz-step-dot {
  width: 28px; height: 28px;
  min-width: 28px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-head);
  font-size: 0.78rem; font-weight: 800;
  border: 2px solid var(--border-strong);
  color: var(--text-muted);
  background: var(--surface);
}
.wiz-step-dot.done {
  background: linear-gradient(135deg, #0891B2, #0D9488);
  border-color: transparent;
  color: #fff;
}
.wiz-step-dot.active {
  border-color: var(--teal-500);
  color: var(--teal-600);
  background: var(--teal-50);
}
.wiz-step-label {
  font-family: var(--font-head);
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--text-muted);
  letter-spacing: 0.01em;
}
.wiz-step-label.active { color: var(--text-primary); }
.wiz-step-line {
  flex: 1;
  height: 2px;
  background: var(--border-strong);
  margin: 0 2px;
}
.wiz-step-line.done { background: var(--teal-500); }

.wiz-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 28px 32px 24px;
  box-shadow: var(--shadow-xs);
  margin-bottom: 16px;
}
.wiz-card-title {
  font-family: var(--font-head) !important;
  font-size: 1.1rem !important;
  font-weight: 800 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.01em !important;
  margin: 0 0 2px !important;
}
.wiz-card-sub {
  font-size: 0.82rem !important;
  color: var(--text-secondary) !important;
  margin: 0 0 20px !important;
}
.wiz-help {
  font-size: 0.74rem !important;
  color: var(--text-muted) !important;
  margin: -10px 0 12px !important;
  line-height: 1.4 !important;
}
.wiz-bmi-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--teal-50);
  border: 1px solid var(--teal-100);
  border-radius: 999px;
  padding: 6px 14px;
  margin-top: 4px;
  font-family: var(--font-head);
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--teal-700);
}

/* ── Metric overrides ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 14px !important;
  box-shadow: var(--shadow-xs) !important;
}
/* ── Mobile Responsiveness ── */
@media (max-width: 768px) {
  .top-bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
  .cri-wrap {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
    padding: 20px;
  }
  .hiw-grid {
    grid-template-columns: 1fr;
  }
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _risk_props(pct: float):
    # 4-tier framework at 25 / 50 / 75, matching the project's documented risk
    # bands (Low / Medium / High / Critical).
    if pct < 25:
        return {'color': '#0D9488', 'bg': '#F0FDFB', 'border': '#99F6E4', 'label': 'Low',      'dot': '#0D9488'}
    elif pct < 50:
        return {'color': '#D97706', 'bg': '#FFFBEB', 'border': '#FCD34D', 'label': 'Medium',   'dot': '#D97706'}
    elif pct < 75:
        return {'color': '#EA580C', 'bg': '#FFF7ED', 'border': '#FED7AA', 'label': 'High',     'dot': '#EA580C'}
    else:
        return {'color': '#DC2626', 'bg': '#FEF2F2', 'border': '#FECACA', 'label': 'Critical', 'dot': '#DC2626'}

def encode_inputs(age, sex, bmi, sbp, dbp, glucose, chol, smoking,
                   height=None, waist_circumference=None, resting_pulse=None,
                   uric_acid=None, cigs_per_day=0.0, prevalent_hyp=None, bp_meds=None):
    # All numeric features are in the canonical units declared in src/schema.py
    # (glucose & cholesterol in mg/dL — every dataset is cleaned to these units).
    # Smoking is encoded as binary because every model was trained on a binary
    # smoke flag (0/1); offering an intermediate value would feed the tree a
    # value it never saw at training time. (alcohol/physical_activity were
    # dropped project-wide: no NHANES module for either is available in this
    # environment, so no shipped model consumes them any more — see
    # scripts/build_hypertension_nhanes.py.)
    #
    # Optional fields (waist, pulse, uric acid, height, hypertension history,
    # BP meds) are left OUT of the dict entirely when the patient doesn't know
    # them — every model imputes a missing feature with its own training
    # median, exactly like a real partial checkup.
    d = {
        'age':               float(age),
        'sex':               1.0 if sex == "Male" else 0.0,
        'bmi':               float(bmi),
        'systolic_bp':       float(sbp),
        'diastolic_bp':      float(dbp),
        'glucose':           float(glucose),
        'cholesterol':       float(chol),
        'smoking':           {'Never': 0.0, 'Former': 0.0, 'Current': 1.0}[smoking],
    }
    if height is not None:
        d['height'] = float(height)
    if waist_circumference is not None:
        d['waist_circumference'] = float(waist_circumference)
    if resting_pulse is not None:
        d['resting_pulse'] = float(resting_pulse)
        d['heartRate'] = float(resting_pulse)  # Framingham (heart model) naming
    if uric_acid is not None:
        d['uric_acid'] = float(uric_acid)
    d['cigs_per_day'] = float(cigs_per_day)
    d['cigsPerDay'] = float(cigs_per_day)      # Framingham (heart model) naming
    if prevalent_hyp is not None:
        d['prevalentHyp'] = float(prevalent_hyp)
    if bp_meds is not None:
        d['BPMeds'] = float(bp_meds)
    return d

def validate_inputs(age, bmi, sbp, dbp, glucose, chol):
    """Return (field, message) clinical-range warnings, driven by the canonical
    ranges declared in src/schema.py (single source of truth)."""
    return range_warnings({
        'age': age, 'bmi': bmi, 'systolic_bp': sbp, 'diastolic_bp': dbp,
        'glucose': glucose, 'cholesterol': chol,
    })

def make_gauge(pct: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            'suffix': '%',
            'font': {'size': 40, 'color': color, 'family': 'Plus Jakarta Sans'},
            'valueformat': '.1f',
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': '#C8CDD6',
                'tickfont':  {'color': '#8896A7', 'size': 10, 'family': 'Inter'},
                'dtick': 25,
            },
            'bar': {'color': color, 'thickness': 0.20},
            'bgcolor': '#F7F8FA',
            'bordercolor': '#E2E5EA',
            'borderwidth': 1,
            'steps': [
                {'range': [0,  25], 'color': '#F0FDFB'},
                {'range': [25, 50], 'color': '#FFFBEB'},
                {'range': [50, 75], 'color': '#FFF7ED'},
                {'range': [75,100], 'color': '#FEF2F2'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 2},
                'thickness': 0.78,
                'value': pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=24, r=24, t=12, b=12),
        height=210,
        font_color='#0F1923',
    )
    return fig

def make_shap_bar(shap_dict: dict, disease: str) -> go.Figure:
    items   = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    feats   = [i[0].replace('_', ' ').title() for i in items]
    vals    = [i[1] for i in items]
    colors  = ['#DC2626' if v > 0 else '#0D9488' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in vals],
        textposition='outside',
        textfont={'size': 10, 'color': '#8896A7', 'family': 'Inter'},
    ))
    fig.update_layout(
        title=dict(
            text=f'Feature Impact — {disease}',
            font={'size': 12, 'family': 'Plus Jakarta Sans', 'color': '#4A5568'},
            x=0,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text='SHAP Value', font={'size': 10, 'family': 'Inter', 'color': '#8896A7'}),
            gridcolor='#E2E5EA',
            zerolinecolor='#C8CDD6',
            tickfont={'size': 10, 'family': 'Inter', 'color': '#8896A7'},
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont={'size': 11, 'family': 'Inter', 'color': '#4A5568'},
        ),
        margin=dict(l=8, r=64, t=40, b=24),
        height=290,
    )
    return fig

def make_roc_chart(cache_data: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"Diabetes": "#0891B2", "CKD": "#7C3AED", "Heart Disease": "#DC2626", "Hypertension": "#D97706"}
    
    for name, data in cache_data.items():
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode='lines',
            name=f"{name} (AUC={data['roc_auc']:.3f})",
            line=dict(color=colors.get(name, "#333"), width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name="Random", line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(gridcolor='#E2E5EA')
    fig.update_yaxes(gridcolor='#E2E5EA')
    return fig

def make_calibration_chart(cache_data: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"Diabetes": "#0891B2", "CKD": "#7C3AED", "Heart Disease": "#DC2626", "Hypertension": "#D97706"}
    
    for name, data in cache_data.items():
        fig.add_trace(go.Scatter(
            x=data["prob_pred"], y=data["prob_true"],
            mode='lines+markers',
            name=name,
            line=dict(color=colors.get(name, "#333"), width=2),
            marker=dict(size=6)
        ))
        
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name="Perfectly Calibrated", line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig.update_layout(
        title="Calibration Curve (Reliability Diagram)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(gridcolor='#E2E5EA')
    fig.update_yaxes(gridcolor='#E2E5EA')
    return fig

def make_whatif_gauge(pct: float, color: str, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={'text': title, 'font': {'size': 14, 'color': '#4A5568'}},
        number={'suffix': '%', 'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'dtick': 25},
            'bar': {'color': color},
            'bgcolor': '#F7F8FA',
        },
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=180,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────────
# `pdata` is a plain dict we write to explicitly at the end of each wizard step.
# We deliberately do NOT rely on widget-key session_state entries surviving
# across reruns where that widget isn't rendered (e.g. Step 1's fields aren't
# rendered while on Step 2) — that persistence is not guaranteed, so every
# cross-step read goes through this dict instead.
if 'wizard_step' not in st.session_state:
    st.session_state.wizard_step = 1
if 'pdata' not in st.session_state:
    st.session_state.pdata = {}
PDATA = st.session_state.pdata

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-name">AarogyaDrishti AI</div>
      <div class="sb-brand-sub">Comorbidity Risk Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.wizard_step == 4:
        st.markdown('<div class="sb-section">Assessment complete</div>', unsafe_allow_html=True)
        if st.button("✎ Edit my information", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
        if st.button("↺ Start new assessment", use_container_width=True):
            st.session_state.pdata = {}
            st.session_state.wizard_step = 1
            st.rerun()
    else:
        st.markdown('<div class="sb-section">Guided assessment</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:var(--text-secondary,#4A5568);line-height:1.5">'
            'Answer 3 short steps in the main panel — demographics, checkup numbers, and '
            'lifestyle — then get your full risk profile. Anything you don\'t know can be skipped.'
            '</p>', unsafe_allow_html=True)


# ── Top Bar ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="top-bar-left">
    <div class="top-bar-icon">🫀</div>
    <div>
      <div class="top-bar-title">AarogyaDrishti AI</div>
      <div class="top-bar-sub">Personalized Health Risk Assessment</div>
    </div>
  </div>
  <div class="top-bar-badge">AI-Powered Clinical Tool</div>
</div>
""", unsafe_allow_html=True)


# ── Wizard step indicator ────────────────────────────────────────────────────────
def render_wizard_steps(current: int):
    labels = ["Demographics", "Checkup Numbers", "Lifestyle & History"]
    parts = ['<div class="wiz-steps">']
    for i, label in enumerate(labels, start=1):
        state = "done" if i < current else ("active" if i == current else "")
        parts.append(f'<div class="wiz-step">'
                      f'<div class="wiz-step-dot {state}">{"✓" if i < current else i}</div>'
                      f'<div class="wiz-step-label {"active" if i == current else ""}">{label}</div>'
                      f'</div>')
        if i < len(labels):
            parts.append(f'<div class="wiz-step-line {"done" if i < current else ""}"></div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


WIZARD_STEP = st.session_state.wizard_step

# ── Step 1: Demographics ────────────────────────────────────────────────────────
if WIZARD_STEP == 1:
    render_wizard_steps(1)
    st.markdown('<div class="wiz-card">', unsafe_allow_html=True)
    st.markdown('<div class="wiz-card-title">Tell us about you</div>', unsafe_allow_html=True)
    st.markdown('<div class="wiz-card-sub">A few basics to start your health assessment.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=PDATA.get('age', 45))
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0,
                                  value=PDATA.get('height', 170.0), format="%.1f")
    with c2:
        sex = st.selectbox("Sex", ["Male", "Female"],
                            index=["Male", "Female"].index(PDATA.get('sex', 'Male')))
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0,
                                  value=PDATA.get('weight', 70.0), format="%.1f")

    bmi = weight / ((height / 100) ** 2)
    st.markdown(f'<div class="wiz-bmi-pill">Your BMI: {bmi:.1f} kg/m²</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    _, nav_r = st.columns([3, 1])
    with nav_r:
        if st.button("Next →", use_container_width=True, type="primary"):
            PDATA.update({'age': age, 'sex': sex, 'height': height, 'weight': weight, 'bmi': bmi})
            st.session_state.wizard_step = 2
            st.rerun()

# ── Step 2: Checkup Numbers ─────────────────────────────────────────────────────
elif WIZARD_STEP == 2:
    age = PDATA.get('age', 45)
    bmi = PDATA.get('bmi', 24.2)
    render_wizard_steps(2)
    st.markdown('<div class="wiz-card">', unsafe_allow_html=True)
    st.markdown('<div class="wiz-card-title">Your latest checkup numbers</div>', unsafe_allow_html=True)
    st.markdown('<div class="wiz-card-sub">From your last blood pressure reading or blood test. Don\'t have a number? Tick "I don\'t know" and we\'ll use a safe population average.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        systolic_bp = st.number_input("Systolic BP (upper number)", min_value=70, max_value=250,
                                       value=PDATA.get('systolic_bp', 120))
        st.markdown('<div class="wiz-help">The top number on a blood pressure reading, e.g. 120. Normal: below 120 mmHg.</div>', unsafe_allow_html=True)

        glucose = st.number_input("Fasting blood glucose (mg/dL)", min_value=50, max_value=400,
                                   value=PDATA.get('glucose', 100))
        st.markdown('<div class="wiz-help">From a fasting blood test. Normal: 70-99 mg/dL; 126+ is diagnostic for diabetes.</div>', unsafe_allow_html=True)

        idk_waist = st.checkbox("I don't know my waist size", value=PDATA.get('idk_waist', False))
        waist_circumference = st.number_input("Waist circumference (cm)", min_value=40.0, max_value=200.0,
                                               value=PDATA.get('waist_circumference', 90.0),
                                               format="%.1f", disabled=idk_waist)
        st.markdown('<div class="wiz-help">Measured around the navel, standing, after breathing out.</div>', unsafe_allow_html=True)

        idk_uric = st.checkbox("I don't know my uric acid level", value=PDATA.get('idk_uric', False))
        uric_acid = st.number_input("Uric acid (mg/dL)", min_value=1.0, max_value=15.0,
                                     value=PDATA.get('uric_acid', 5.0),
                                     format="%.1f", disabled=idk_uric)
        st.markdown('<div class="wiz-help">From a blood test, if you have one. Normal: 3.5-7.2 (men), 2.6-6.0 (women) mg/dL.</div>', unsafe_allow_html=True)

    with c2:
        diastolic_bp = st.number_input("Diastolic BP (lower number)", min_value=40, max_value=150,
                                        value=PDATA.get('diastolic_bp', 80))
        st.markdown('<div class="wiz-help">The bottom number on a blood pressure reading, e.g. 80. Normal: below 80 mmHg.</div>', unsafe_allow_html=True)

        cholesterol = st.number_input("Total cholesterol (mg/dL)", min_value=100, max_value=400,
                                       value=PDATA.get('cholesterol', 190))
        st.markdown('<div class="wiz-help">From a blood test. Normal: below 200 mg/dL.</div>', unsafe_allow_html=True)

        idk_pulse = st.checkbox("I don't know my resting pulse", value=PDATA.get('idk_pulse', False))
        resting_pulse = st.number_input("Resting pulse (bpm)", min_value=30, max_value=200,
                                         value=PDATA.get('resting_pulse', 72), disabled=idk_pulse)
        st.markdown('<div class="wiz-help">Count your heartbeats for 60 seconds while sitting still. Normal: 60-100 bpm.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    input_warnings = validate_inputs(age, bmi, systolic_bp, diastolic_bp, glucose, cholesterol)
    if input_warnings:
        st.markdown('<div style="background:#FFFBEB;border:1px solid #FCD34D;border-left:3px solid #D97706;'
                     'border-radius:10px;padding:12px 16px;margin-bottom:16px">'
                     '<div style="font-family:var(--font-head);font-size:0.78rem;font-weight:700;'
                     'color:#92400E;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:6px">'
                     '⚠ Clinical Range Warnings</div>', unsafe_allow_html=True)
        for field, msg in input_warnings:
            st.markdown(f'<div style="font-size:0.82rem;color:#78350F;margin:3px 0">'
                        f'<strong>{field}:</strong> {msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    step2_vals = dict(systolic_bp=systolic_bp, diastolic_bp=diastolic_bp, glucose=glucose, cholesterol=cholesterol,
                       idk_waist=idk_waist, waist_circumference=waist_circumference,
                       idk_pulse=idk_pulse, resting_pulse=resting_pulse,
                       idk_uric=idk_uric, uric_acid=uric_acid)

    nav_l, _, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("← Back", use_container_width=True):
            PDATA.update(step2_vals)
            st.session_state.wizard_step = 1
            st.rerun()
    with nav_r:
        if st.button("Next →", use_container_width=True, type="primary"):
            PDATA.update(step2_vals)
            st.session_state.wizard_step = 3
            st.rerun()

# ── Step 3: Lifestyle & History ─────────────────────────────────────────────────
elif WIZARD_STEP == 3:
    render_wizard_steps(3)
    st.markdown('<div class="wiz-card">', unsafe_allow_html=True)
    st.markdown('<div class="wiz-card-title">Lifestyle & medical history</div>', unsafe_allow_html=True)
    st.markdown('<div class="wiz-card-sub">Last step — these help us personalize your risk estimate.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        smoking = st.radio("Smoking status", ["Never", "Former", "Current"],
                            index=["Never", "Former", "Current"].index(PDATA.get('smoking', 'Never')),
                            horizontal=True)
        cigs_per_day = 0.0
        if smoking == "Current":
            cigs_per_day = st.slider("Cigarettes per day", 1, 60, int(PDATA.get('cigs_per_day', 10) or 10))

    with c2:
        prevalent_hyp_choice = st.radio("Diagnosed with high blood pressure before?", ["No", "Yes", "Don't know"],
                                         index=["No", "Yes", "Don't know"].index(PDATA.get('prevalent_hyp_choice', "Don't know")))
        bp_meds_choice = st.radio("Currently on blood pressure medication?", ["No", "Yes"],
                                   index=["No", "Yes"].index(PDATA.get('bp_meds_choice', 'No')))

    st.markdown('</div>', unsafe_allow_html=True)

    step3_vals = dict(smoking=smoking, cigs_per_day=cigs_per_day,
                       prevalent_hyp_choice=prevalent_hyp_choice, bp_meds_choice=bp_meds_choice)

    nav_l, _, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("← Back", use_container_width=True):
            PDATA.update(step3_vals)
            st.session_state.wizard_step = 2
            st.rerun()
    with nav_r:
        if st.button("Calculate Risk Profile →", use_container_width=True, type="primary"):
            PDATA.update(step3_vals)
            st.session_state.wizard_step = 4
            st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────────
else:
    age          = PDATA.get('age', 45)
    sex          = PDATA.get('sex', 'Male')
    height       = PDATA.get('height', 170.0)
    weight       = PDATA.get('weight', 70.0)
    bmi          = weight / ((height / 100) ** 2)
    systolic_bp  = PDATA.get('systolic_bp', 120)
    diastolic_bp = PDATA.get('diastolic_bp', 80)
    glucose      = PDATA.get('glucose', 100)
    cholesterol  = PDATA.get('cholesterol', 190)
    smoking      = PDATA.get('smoking', 'Never')
    cigs_per_day = PDATA.get('cigs_per_day', 0.0) if smoking == "Current" else 0.0

    waist_circumference = None if PDATA.get('idk_waist') else PDATA.get('waist_circumference')
    resting_pulse        = None if PDATA.get('idk_pulse') else PDATA.get('resting_pulse')
    uric_acid             = None if PDATA.get('idk_uric') else PDATA.get('uric_acid')
    prevalent_hyp_choice  = PDATA.get('prevalent_hyp_choice', "Don't know")
    prevalent_hyp         = {"Yes": 1.0, "No": 0.0}.get(prevalent_hyp_choice)  # None if "Don't know"
    bp_meds                = {"Yes": 1.0, "No": 0.0}[PDATA.get('bp_meds_choice', 'No')]

    # ── Input validation ──
    input_warnings = validate_inputs(age, bmi, systolic_bp, diastolic_bp, glucose, cholesterol)
    if input_warnings:
        st.markdown("""
        <div style="background:#FFFBEB;border:1px solid #FCD34D;border-left:3px solid #D97706;
             border-radius:10px;padding:12px 16px;margin-bottom:16px">
          <div style="font-family:var(--font-head);font-size:0.78rem;font-weight:700;
               color:#92400E;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:6px">
            ⚠ Clinical Range Warnings
          </div>
        """, unsafe_allow_html=True)
        for field, msg in input_warnings:
            st.markdown(f"""
          <div style="font-size:0.82rem;color:#78350F;margin:3px 0">
            <strong>{field}:</strong> {msg}
          </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    patient_data = encode_inputs(
        age, sex, bmi, systolic_bp, diastolic_bp,
        glucose, cholesterol, smoking,
        height=height, waist_circumference=waist_circumference, resting_pulse=resting_pulse,
        uric_acid=uric_acid, cigs_per_day=cigs_per_day, prevalent_hyp=prevalent_hyp, bp_meds=bp_meds,
    )

    with st.spinner("Running chained model predictions & SHAP analysis…"):
        try:
            profile   = get_full_risk_profile(patient_data)
            dia_pct   = round(profile['diabetes_risk']     * 100, 1)
            ckd_pct   = round(profile['ckd_risk']          * 100, 1)
            ht_pct    = round(profile['hypertension_risk'] * 100, 1)
            heart_pct = round(profile['heart_risk']        * 100, 1)
            cri_pct   = round(profile['cri']               * 100, 1)

            cri_props = _risk_props(cri_pct)

            # SHAP
            patient_df = pd.DataFrame([patient_data])
            patient_df_chain = patient_df.copy()
            patient_df_chain['diabetes_risk'] = profile['diabetes_risk']
            patient_df_chain['ckd_risk']      = profile['ckd_risk']

            def _prep_for_shap(df, raw_features, engineer_fn, medians):
                """Mirror each model's own predict_risk(): fill any raw feature
                the wizard left out (e.g. prevalentHyp when the patient answered
                "Don't know") with the model's training median, THEN derive the
                engineered columns (pulse pressure, interaction terms, ...) —
                encode_inputs()'s raw output alone doesn't have those."""
                df = df.copy()
                for col in raw_features:
                    if col not in df.columns:
                        df[col] = medians[col]
                    else:
                        df[col] = df[col].fillna(medians[col])
                return engineer_fn(df)

            shap_res = {}
            try:
                dm = load_diabetes();  ckm = load_ckd()
                hm = load_heart();     htm = load_hypertension()
                dia_df   = _prep_for_shap(patient_df,       RAW_DIABETES_FEATURES, engineer_diabetes, dm['medians'])
                ckd_df   = _prep_for_shap(patient_df,       RAW_CKD_FEATURES,      engineer_ckd,      ckm['medians'])
                heart_df = _prep_for_shap(patient_df_chain, RAW_HEART_FEATURES,    engineer_heart,    hm['medians'])
                ht_df    = _prep_for_shap(patient_df_chain, RAW_HT_FEATURES,       engineer_hypertension, htm['medians'])
                shap_res['Diabetes']      = explain_prediction(dm['model'],  dia_df[dm['features']])
                shap_res['CKD']           = explain_prediction(ckm['model'], ckd_df[ckm['features']])
                shap_res['Heart Disease'] = explain_prediction(hm['model'],  heart_df[hm['features']])
                shap_res['Hypertension']  = explain_prediction(htm['model'], ht_df[htm['features']])
            except Exception:
                shap_res = {}

            # ════════════════════════════════════
            # 1. CRI SECTION
            # ════════════════════════════════════
            gauge_col, info_col = st.columns([1, 1.5], gap="large")

            with gauge_col:
                st.plotly_chart(make_gauge(cri_pct, cri_props['color']), use_container_width=True)

            with info_col:
                desc = {
                    "Low":      "This patient presents a low overall comorbidity burden. Routine preventive care and annual screenings are recommended.",
                    "Medium":   "This patient shows emerging comorbidity risk with modifiable factors present. Targeted lifestyle intervention and monitoring are advisable.",
                    "High":     "This patient presents high comorbidity risk with multiple concurrent factors. Clinical work-up and an intervention plan are strongly recommended.",
                    "Critical": "This patient presents a critical comorbidity burden. Immediate clinical review, further diagnostics, and active management are strongly recommended.",
                }[cri_props['label']]

                st.markdown(f"""
                <div class="cri-wrap" style="border-top: 3px solid {cri_props['color']}">
                  <div>
                    <div class="cri-label">Comorbidity Risk Index</div>
                    <div class="cri-number" style="color:{cri_props['color']}">{cri_pct}%</div>
                    <div style="margin-top: 10px">
                      <span class="status-badge"
                        style="background:{cri_props['bg']};border-color:{cri_props['border']};color:{cri_props['color']}">
                        <span class="status-dot" style="background:{cri_props['dot']}"></span>
                        {cri_props['label']} Risk
                      </span>
                    </div>
                    <div class="cri-desc">{desc}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # ════════════════════════════════════
            # 2. DISEASE RISK CARDS
            # ════════════════════════════════════
            st.markdown("""
            <div class="sec-head">
              <div class="sec-head-title">Disease Risk Breakdown</div>
              <div class="sec-head-sub">Chained model predictions — upstream risks inform downstream estimates</div>
            </div>
            """, unsafe_allow_html=True)

            # Model accuracy annotations per disease (5-fold CV, checkup-safe;
            # Heart & Hypertension are the shipped *chained* models).
            model_acc = {
                "Diabetes":      ("81.3%", "0.864"),
                "CKD":           ("78.1%", "0.777"),
                "Heart Disease": ("74.2%", "0.713"),
                "Hypertension":  ("73.5%", "0.811"),
            }
            ckd_footnote = True

            diseases = [
                ("Diabetes",      dia_pct,   "Upstream", "#0891B2"),
                ("CKD",           ckd_pct,   "Upstream", "#7C3AED"),
                ("Heart Disease", heart_pct, "Downstream", "#DC2626"),
                ("Hypertension",  ht_pct,    "Downstream", "#D97706"),
            ]

            cols = st.columns(4, gap="small")
            for col, (name, pct, chain, accent) in zip(cols, diseases):
                p = _risk_props(pct)
                acc, auc = model_acc[name]
                ckd_note = (
                    '<div style="font-size:0.65rem;color:#92400E;background:#FFFBEB;'
                    'border:1px solid #FCD34D;border-radius:4px;padding:3px 6px;margin-top:8px;line-height:1.4">'
                    'Checkup-safe only (AUC 0.78). Adding a serum-creatinine lab → AUC 0.82.'
                    '</div>'
                ) if name == "CKD" else ""
                with col:
                    st.markdown(f"""
                    <div class="dc-card">
                      <div class="dc-card-accent" style="background:{accent}"></div>
                      <div class="dc-card-name">{name}</div>
                      <div class="dc-card-value" style="color:{p['color']}">{pct}%</div>
                      <div class="dc-card-track">
                        <div class="dc-card-fill" style="width:{pct}%;background:{p['color']}40;
                             border-right: 2px solid {p['color']}"></div>
                      </div>
                      <span class="status-badge" style="background:{p['bg']};border-color:{p['border']};color:{p['color']};font-size:0.65rem">
                        <span class="status-dot" style="background:{p['dot']};width:5px;height:5px"></span>
                        {p['label']}
                      </span>
                      {ckd_note}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.divider()

            # ════════════════════════════════════
            # 3. PATIENT INSIGHTS
            # ════════════════════════════════════
            st.markdown("""
            <div class="sec-head">
              <div class="sec-head-title">Personalized Insights</div>
              <div class="sec-head-sub">What's driving your health scores and what you can do about it.</div>
            </div>
            """, unsafe_allow_html=True)

            from src.explainability.generator import InsightGenerator
            insight_generator = InsightGenerator()
            if shap_res:
                patient_dict = patient_df_chain.iloc[0].to_dict()
                for disease, sd in shap_res.items():
                    if sd:
                        expl = insight_generator.generate(sd, patient_data=patient_dict)
                        st.markdown(f"""
                        <div class="insight-card" style="margin-bottom:12px;">
                          <p>
                            <strong style="color:var(--teal-600)">For {disease}:</strong><br>
                            {expl}
                          </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("SHAP explanations could not be computed.")

            st.divider()

            # ════════════════════════════════════
            # 4. INPUT SUMMARY
            # ════════════════════════════════════
            st.markdown("""
            <div class="sec-head">
              <div class="sec-head-title">Patient Input Summary</div>
              <div class="sec-head-sub">Values used in this prediction run</div>
            </div>
            """, unsafe_allow_html=True)

            summary_df = pd.DataFrame({
                "Feature":  ["Age", "Sex", "Height", "Weight", "BMI (Calculated)", "Upper BP", "Lower BP",
                             "Blood Sugar", "Total Cholesterol", "Waist Circumference", "Resting Pulse", "Uric Acid",
                             "Smoking", "Cigarettes/day",
                             "Prior Hypertension Dx", "On BP Medication"],
                "Value":    [str(age), str(sex), f"{height} cm", f"{weight} kg", f"{bmi:.1f}", f"{systolic_bp} mmHg", f"{diastolic_bp} mmHg",
                             f"{glucose} mg/dL", f"{cholesterol} mg/dL",
                             f"{waist_circumference} cm" if waist_circumference is not None else "Not provided",
                             f"{resting_pulse} bpm" if resting_pulse is not None else "Not provided",
                             f"{uric_acid} mg/dL" if uric_acid is not None else "Not provided",
                             str(smoking), f"{cigs_per_day:.0f}" if smoking == "Current" else "—",
                             prevalent_hyp_choice, PDATA.get('bp_meds_choice', 'No')],
                "Category": ["Demographic", "Demographic", "Vitals", "Vitals", "Vitals", "Vitals", "Vitals",
                             "Vitals", "Vitals", "Vitals", "Vitals", "Vitals",
                             "Lifestyle", "Lifestyle",
                             "History", "History"],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.divider()

            # ════════════════════════════════════
            # 5. MODEL EVIDENCE PANEL (Phase 2)
            # ════════════════════════════════════
            with st.expander("🔬 For Doctors & Researchers: Technical Details", expanded=False):
                st.markdown("""
                <div class="sec-head" style="margin-bottom:12px">
                  <div class="sec-head-title">Model Performance — Checkup-Safe Feature Set</div>
                  <div class="sec-head-sub">5-fold cross-validation with SMOTE on training splits · All models use only routine checkup features</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Metrics table ──
                metrics_df = pd.DataFrame({
                    "Model":      ["Diabetes", "CKD", "Heart Disease", "Hypertension"],
                    "Accuracy":   ["81.3%", "78.1%", "74.2%", "73.5%"],
                    "ROC AUC":    ["0.864", "0.777", "0.713", "0.811"],
                    "F1-Score":   ["0.523", "0.492", "0.360", "0.672"],
                    "Dataset Size":["7,912", "5,154", "4,240", "8,139"],
                    "Feature Set":["Checkup-safe, 14 features (NHANES 2021-2023)", "Checkup-safe, 13 features (NHANES 2017-2018)", "Chained checkup-safe, 16 features", "Chained checkup-safe, 16 features (NHANES 2021-2023)"],
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                
                # ── ROC & Calibration Curves ──
                st.markdown("""
                <div class="sec-head" style="margin-bottom:12px">
                  <div class="sec-head-title">Validation Curves — Held-out 20% Test Split</div>
                  <div class="sec-head-sub">A fresh model with the shipped configuration is trained on the 80% train split and scored on the untouched 20% test split (no leakage). Discrimination (ROC) and reliability (Calibration).</div>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    eval_cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "eval_cache.pkl")
                    eval_cache = joblib.load(eval_cache_path)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(make_roc_chart(eval_cache), use_container_width=True)
                    with col2:
                        st.plotly_chart(make_calibration_chart(eval_cache), use_container_width=True)
                except Exception as e:
                    st.warning("Validation curves unavailable. Run `generate_eval_cache.py` first.")
                    
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                # ── Chaining Delta table ──
                st.markdown("""
                <div class="sec-head" style="margin-bottom:12px">
                  <div class="sec-head-title">Chained vs. Isolated Prediction Comparison</div>
                  <div class="sec-head-sub">Ablation on the shipped models: does adding upstream Diabetes + CKD risk as features change 5-fold CV performance? (Heart & HTN are the models actually deployed here.)</div>
                </div>
                """, unsafe_allow_html=True)

                # Deployed-model ablation (5-fold CV, from reports/model_metrics.json).
                chaining_df = pd.DataFrame({
                    "Deployed Model":     ["Heart Disease", "Hypertension"],
                    "Dataset":            ["heart_clean / Framingham (n=4,240)", "hypertension_nhanes_clean (n=8,139)"],
                    "Isolated Acc":       ["78.40%", "73.88%"],
                    "Chained Acc":        ["74.20%", "73.51%"],
                    "Δ Accuracy":         ["-4.20%", "-0.37%"],
                    "Isolated AUC":       ["0.707", "0.814"],
                    "Chained AUC":        ["0.713", "0.811"],
                })
                st.dataframe(chaining_df, use_container_width=True, hide_index=True)

                st.markdown("""
                <div class="insight-card" style="margin-top:14px">
                  <p>
                    <strong style="color:var(--teal-600)">Key Finding —</strong>
                    Chaining moves the two deployed downstream models in different directions on accuracy vs.
                    AUC. For Heart Disease, adding the upstream Diabetes/CKD risk scores improves ROC AUC
                    (+0.006) — the model discriminates cases better — but costs 4.2 accuracy points, because the
                    chained model (trained with SMOTE) shifts more borderline cases past the 0.5 threshold into
                    a positive prediction, trading precision for recall on a ~15% positive dataset. Hypertension
                    is slightly negative on both metrics (Accuracy −0.37pp, AUC −0.0027) — after its P1 dataset
                    swap to NHANES doctor-diagnosed hypertension, its own checkup-safe features already carry
                    most of the signal, so chaining doesn't have much to add. A wider ablation on two independent
                    comorbidity cohorts (see
                    <code>reports/chaining_results.md</code>) shows the same pattern of the chained variant
                    trading some accuracy for better ranking/discrimination. Chaining is a targeted prior, not a
                    blanket accuracy win — use AUC, not accuracy alone, to judge whether chaining is worth it for
                    a given model.
                  </p>
                </div>

                <div style="background:#F0FDF4;border:1px solid #86EFAC;border-left:3px solid #16A34A;
                     border-radius:10px;padding:12px 16px;margin-top:10px">
                  <div style="font-family:var(--font-head);font-size:0.75rem;font-weight:700;
                       color:#14532D;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:6px">
                    CRI Formula — Weight Rationale
                  </div>
                  <div style="font-size:0.82rem;color:#166534;line-height:1.7">
                    <strong>CRI = 0.30×P(Diabetes) + 0.20×P(CKD) + 0.25×P(Heart) + 0.25×P(HTN)</strong><br>
                    + 0.15×P(Diabetes)×P(Heart) &nbsp;·&nbsp; Diabetics are 2–4× more likely to develop CAD (ADA/ACC meta-analyses)<br>
                    + 0.10×P(HTN)×P(Heart) &nbsp;·&nbsp; Vascular strain amplifier<br>
                    + 0.05×P(Diabetes)×P(HTN) &nbsp;·&nbsp; Overlapping metabolic syndrome pathway
                  </div>
                </div>

                <div style="background:#FEF2F2;border:1px solid #FECACA;border-left:3px solid #DC2626;
                     border-radius:10px;padding:12px 16px;margin-top:10px">
                  <div style="font-family:var(--font-head);font-size:0.75rem;font-weight:700;
                       color:#7F1D1D;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:6px">
                    ⚠ Limitations & Ethics
                  </div>
                  <div style="font-size:0.82rem;color:#991B1B;line-height:1.7">
                    This is a <strong>screening tool, not a diagnostic instrument.</strong>
                    Results must be reviewed by a qualified clinician before any clinical decision.
                    <strong>Diabetes model:</strong> trained on <strong>NHANES 2021-2023</strong>
                    (7,912 adults, both sexes, ~15.5% positive), 14 checkup-safe features (added waist
                    circumference, resting pulse, uric acid, smoking dose, pulse pressure, and an
                    age×glucose interaction term) with monotonic constraints ensuring clinically correct
                    feature directions (higher glucose always increases risk).
                    <strong>CKD model:</strong> trained on NHANES 2017-2018 (5,154 adults, both sexes),
                    13 checkup-safe features with the same expansion pattern; a full lab panel is still
                    needed for clinical CKD diagnosis.
                    <strong>Heart Disease model:</strong> 16 chained features, adding resting heart rate,
                    smoking dose, prior-hypertension diagnosis, and BP medication status on top of the
                    original checkup-safe set — the latter two require the patient to answer two yes/no
                    questions, not a lab test.
                    <strong>Hypertension model:</strong> also moved to NHANES 2021-2023 (8,139 adults, both
                    sexes, ~36.3% positive), replacing the cardio/Kaggle dataset's broad self-reported
                    cardiovascular-disease flag with a doctor-diagnosed-hypertension label, and its ordinal
                    (1–3) cholesterol/glucose with continuous lab values. 16 chained features. No NHANES
                    alcohol-use or physical-activity module was available to rebuild those two fields — since
                    no shipped model consumed them any more once they were dropped from this dataset, they
                    were removed from the assessment entirely rather than kept as questions whose answers do
                    nothing.
                    Any field the patient doesn't know is simply omitted — each model falls back to its own
                    training-population median for that feature, the same as it would for a partial
                    real-world checkup.
                    <strong>Hyperparameters:</strong> all four models' XGBoost settings (tree count, depth,
                    learning rate, regularization) were tuned with a 60-trial Optuna search per model against
                    5-fold CV ROC AUC; monotonic constraints are never tuned, since they encode a clinical
                    correctness requirement rather than a performance knob.
                    <strong>Chaining:</strong> the upstream Diabetes/CKD risk features fed to the Heart
                    and Hypertension models were, at training time, computed on datasets missing some
                    inputs (median-imputed), so those features carry less signal in training than at
                    live inference — a cross-dataset limitation, not a validated end-to-end pipeline.
                    All models use monotonic constraints to prevent clinically inverted predictions.
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.divider()
            
            # ════════════════════════════════════
            # 6. WHAT-IF RISK SIMULATOR (Phase 3)
            # ════════════════════════════════════
            st.markdown("""
            <div class="sec-head">
              <div class="sec-head-title">🎛️ What Happens If I Change My Habits?</div>
              <div class="sec-head-sub">Adjust your weight and lifestyle below to see how it improves your health score.</div>
            </div>
            """, unsafe_allow_html=True)
            
            sim_col1, sim_col2 = st.columns([1, 1.5], gap="large")
            
            with sim_col1:
                st.markdown("**Modifiable Factors**")
                sim_weight = st.slider("Weight (kg)", 30.0, 200.0, float(weight), 1.0, key="sim_weight")
                sim_bmi = sim_weight / ((height / 100) ** 2)
                sim_glucose = st.slider("Blood Sugar", 70, 300, int(glucose), 5, key="sim_glucose")
                sim_sbp = st.slider("Upper Blood Pressure", 90, 200, int(systolic_bp), 2, key="sim_sbp")
                sim_smoking = st.selectbox("Smoking", ["Never", "Former", "Current"], index=["Never", "Former", "Current"].index(smoking), key="sim_smoking")

            with sim_col2:
                # Re-run prediction with simulated inputs
                sim_cigs = PDATA.get('cigs_per_day', 0.0) if sim_smoking == "Current" else 0.0
                sim_data = encode_inputs(
                    age, sex, sim_bmi, sim_sbp, diastolic_bp,
                    sim_glucose, cholesterol, sim_smoking,
                    height=height, waist_circumference=waist_circumference, resting_pulse=resting_pulse,
                    uric_acid=uric_acid, cigs_per_day=sim_cigs, prevalent_hyp=prevalent_hyp, bp_meds=bp_meds,
                )
                sim_profile = get_full_risk_profile(sim_data)
                sim_cri_pct = round(sim_profile['cri'] * 100, 1)
                
                delta = sim_cri_pct - cri_pct
                
                delta_color = "#16A34A" if delta < 0 else ("#DC2626" if delta > 0 else "#64748B")
                delta_sign = "+" if delta > 0 else ""
                
                st.plotly_chart(make_gauge(sim_cri_pct, _risk_props(sim_cri_pct)['color']), use_container_width=True, key="what_if_gauge")
                
                st.markdown(f"""
                <div style="text-align:center; padding: 10px; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0;">
                  <span style="font-size: 0.9rem; color: #475569;">Baseline CRI: <strong>{cri_pct}%</strong></span> &nbsp;|&nbsp; 
                  <span style="font-size: 0.9rem; color: #475569;">Simulated Change: 
                    <strong style="color: {delta_color}; font-size: 1.1rem;">{delta_sign}{delta:.1f}%</strong>
                  </span>
                </div>
                """, unsafe_allow_html=True)
                

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

