import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chaining.cri import get_full_risk_profile
from src.explainability.shap_engine import explain_prediction
import shap
import matplotlib.pyplot as plt
from src.models.diabetes_model import _load_model_data as load_diabetes
from src.models.ckd_model import _load_model_data as load_ckd
from src.models.heart_model import _load_model_data as load_heart
from src.models.hypertension_model import _load_model_data as load_hypertension

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

/* ── Metric overrides ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 14px !important;
  box-shadow: var(--shadow-xs) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _risk_props(pct: float):
    if pct < 30:
        return {
            'color':  '#0D9488',
            'bg':     '#F0FDFB',
            'border': '#99F6E4',
            'label':  'Low',
            'dot':    '#0D9488',
        }
    elif pct < 60:
        return {
            'color':  '#D97706',
            'bg':     '#FFFBEB',
            'border': '#FCD34D',
            'label':  'Moderate',
            'dot':    '#D97706',
        }
    else:
        return {
            'color':  '#DC2626',
            'bg':     '#FEF2F2',
            'border': '#FECACA',
            'label':  'High',
            'dot':    '#DC2626',
        }

def encode_inputs(age, sex, bmi, sbp, dbp, glucose, chol, smoking, alcohol, pa, fh):
    return {
        'age':               float(age),
        'sex':               1.0 if sex == "Male" else 0.0,
        'bmi':               float(bmi),
        'systolic_bp':       float(sbp),
        'diastolic_bp':      float(dbp),
        'glucose':           float(glucose),
        'cholesterol':       float(chol),
        'smoking':           {'Never': 0.0, 'Former': 0.5, 'Current': 1.0}[smoking],
        'alcohol':           {'None': 0.0, 'Occasional': 0.5, 'Frequent': 1.0}[alcohol],
        'physical_activity': {'Low': 0.0, 'Moderate': 0.5, 'High': 1.0}[pa],
        'family_history':    1.0 if fh == "Yes" else 0.0,
    }

def validate_inputs(age, bmi, sbp, dbp, glucose, chol):
    """Return list of (field, message) clinical range warnings."""
    warnings = []
    if not (1 <= age <= 110):
        warnings.append(("Age", f"{age} is outside the expected range 1–110 years."))
    if not (13.0 <= bmi <= 60.0):
        warnings.append(("BMI", f"{bmi:.1f} is outside the physiologically plausible range 13–60."))
    if not (70 <= sbp <= 220):
        warnings.append(("Systolic BP", f"{sbp} mmHg is outside the expected range 70–220."))
    if not (40 <= dbp <= 140):
        warnings.append(("Diastolic BP", f"{dbp} mmHg is outside the expected range 40–140."))
    if sbp <= dbp:
        warnings.append(("BP relationship", f"Systolic BP ({sbp}) must be greater than Diastolic BP ({dbp})."))
    if not (50 <= glucose <= 600):
        warnings.append(("Fasting Glucose", f"{glucose} mg/dL is outside the expected range 50–600."))
    if not (100 <= chol <= 500):
        warnings.append(("Cholesterol", f"{chol} mg/dL is outside the expected range 100–500."))
    return warnings

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
                {'range': [0,  30], 'color': '#F0FDFB'},
                {'range': [30, 60], 'color': '#FFFBEB'},
                {'range': [60,100], 'color': '#FEF2F2'},
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
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-name">AarogyaDrishti AI</div>
      <div class="sb-brand-sub">Comorbidity Risk Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        st.markdown('<div class="sb-section">Demographics</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])

        st.markdown('<div class="sb-section">Clinical Vitals</div>', unsafe_allow_html=True)
        height      = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, format="%.1f")
        weight      = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, format="%.1f")
        bmi         = weight / ((height / 100) ** 2)
        systolic_bp = st.number_input("Upper Blood Pressure (Systolic)",  min_value=70, max_value=250, value=120)
        diastolic_bp= st.number_input("Lower Blood Pressure (Diastolic)", min_value=40, max_value=150, value=80)
        glucose     = st.number_input("Blood Sugar (Fasting)", min_value=50, max_value=400, value=100)
        cholesterol = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=190)

        st.markdown('<div class="sb-section">Lifestyle & History</div>', unsafe_allow_html=True)
        smoking  = st.selectbox("Smoking Status",        ["Never", "Former", "Current"])
        alcohol  = st.selectbox("Alcohol Consumption",   ["None", "Occasional", "Frequent"])
        phys_act = st.selectbox("Physical Activity",     ["Low", "Moderate", "High"])
        fam_hist = st.selectbox("Family History of Chronic Illness", ["No", "Yes"])

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        submit = st.form_submit_button("Calculate Risk Profile", use_container_width=True)
        if submit:
            st.session_state.submitted = True



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


# ── Empty State ───────────────────────────────────────────────────────────────────
if not st.session_state.get('submitted', False):
    st.markdown("""
    <div class="empty-wrap">
      <div class="empty-icon">🩺</div>
      <h3>No health data yet</h3>
      <p>
        Enter your health details in the sidebar and click<br>
        <strong>Calculate Risk Profile</strong> to generate an AI-powered health assessment.
      </p>
    </div>

    <div class="hiw-grid">
      <div class="hiw-card">
        <div class="hiw-num">1</div>
        <h4>Upstream Risks</h4>
        <p>Diabetes and CKD risks are independently predicted from patient vitals</p>
      </div>
      <div class="hiw-card">
        <div class="hiw-num">2</div>
        <h4>Risk Chaining</h4>
        <p>Upstream probabilities are fed as features into Heart Disease and Hypertension models</p>
      </div>
      <div class="hiw-card">
        <div class="hiw-num">3</div>
        <h4>CRI Computation</h4>
        <p>A weighted Comorbidity Risk Index is calculated including interaction penalties</p>
      </div>
      <div class="hiw-card">
        <div class="hiw-num">4</div>
        <h4>SHAP Explainability</h4>
        <p>Feature-level SHAP values show what drives each disease risk score</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────────
else:
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
        glucose, cholesterol, smoking, alcohol, phys_act, fam_hist,
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

            shap_res = {}
            shap_waterfalls = {}
            
            def get_shap_explanation(model, patient_row: pd.DataFrame):
                explainer = shap.Explainer(model)
                shap_values = explainer(patient_row)
                if len(shap_values.shape) == 3: # For multiclass
                    return shap_values[0, :, 1]
                return shap_values[0]
                
            try:
                dm = load_diabetes();  ckm = load_ckd()
                hm = load_heart();     htm = load_hypertension()
                shap_res['Diabetes']      = explain_prediction(dm['model'],  patient_df[dm['features']])
                shap_res['CKD']           = explain_prediction(ckm['model'], patient_df[ckm['features']])
                shap_res['Heart Disease'] = explain_prediction(hm['model'],  patient_df_chain[hm['features']])
                shap_res['Hypertension']  = explain_prediction(htm['model'], patient_df_chain[htm['features']])
                
                shap_waterfalls['Diabetes'] = get_shap_explanation(dm['model'], patient_df[dm['features']])
                shap_waterfalls['CKD'] = get_shap_explanation(ckm['model'], patient_df[ckm['features']])
                shap_waterfalls['Heart Disease'] = get_shap_explanation(hm['model'], patient_df_chain[hm['features']])
                shap_waterfalls['Hypertension'] = get_shap_explanation(htm['model'], patient_df_chain[htm['features']])
            except Exception:
                shap_res = {}
                shap_waterfalls = {}

            # ════════════════════════════════════
            # 1. CRI SECTION
            # ════════════════════════════════════
            gauge_col, info_col = st.columns([1, 1.5], gap="large")

            with gauge_col:
                st.plotly_chart(make_gauge(cri_pct, cri_props['color']), use_container_width=True)

            with info_col:
                desc = {
                    "Low":      "This patient presents a low overall comorbidity burden. Routine preventive care and annual screenings are recommended.",
                    "Moderate": "This patient shows moderate comorbidity risk. A specialist review and targeted lifestyle interventions are advisable.",
                    "High":     "This patient presents high comorbidity risk. Immediate clinical review, further diagnostics, and an intervention plan are strongly recommended.",
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

            # Model accuracy annotations per disease
            model_acc = {
                "Diabetes":      ("76.4%", "0.825"),
                "CKD":           ("79.0%", "0.892"),
                "Heart Disease": ("88.6%", "0.963"),
                "Hypertension":  ("73.6%", "0.803"),
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
                    '⚠ Checkup-safe features only (79% acc). Full lab panel: 97.8%.'
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
                             "Blood Sugar", "Total Cholesterol",
                             "Smoking", "Alcohol", "Physical Activity", "Family History"],
                "Value":    [str(age), str(sex), f"{height} cm", f"{weight} kg", f"{bmi:.1f}", f"{systolic_bp} mmHg", f"{diastolic_bp} mmHg",
                             f"{glucose} mg/dL", f"{cholesterol} mg/dL",
                             str(smoking), str(alcohol), str(phys_act), str(fam_hist)],
                "Category": ["Demographic", "Demographic", "Vitals", "Vitals", "Vitals", "Vitals", "Vitals",
                             "Vitals", "Vitals",
                             "Lifestyle", "Lifestyle", "Lifestyle", "History"],
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
                    "Accuracy":   ["76.4%", "79.0%", "88.6%", "73.6%"],
                    "ROC AUC":    ["0.825", "0.892", "0.963", "0.803"],
                    "F1-Score":   ["0.683", "0.824", "0.886", "0.725"],
                    "Dataset Size":["768", "400", "1,025", "70,000"],
                    "Feature Set":["Checkup-safe", "⚠ Reduced (lab tests removed)", "Checkup-safe", "Checkup-safe"],
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                
                # ── ROC & Calibration Curves ──
                st.markdown("""
                <div class="sec-head" style="margin-bottom:12px">
                  <div class="sec-head-title">Validation Curves — Held-out 20% Test Split</div>
                  <div class="sec-head-sub">Model discrimination (ROC) and reliability (Calibration)</div>
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
                  <div class="sec-head-sub">Does feeding upstream Diabetes + CKD risk improve downstream predictions?</div>
                </div>
                """, unsafe_allow_html=True)

                chaining_df = pd.DataFrame({
                    "Dataset":            ["Heart+HTN Dataset", "Heart+HTN Dataset", "BRFSS Dataset", "BRFSS Dataset"],
                    "Predicts":           ["Heart Disease", "Hypertension", "Heart Disease", "Hypertension"],
                    "Isolated Acc":       ["88.67%", "86.05%", "81.27%", "70.36%"],
                    "Chained Acc":        ["88.75%", "85.85%", "82.67%", "70.36%"],
                    "Δ Accuracy":         ["+0.08%", "-0.20%", "+1.40% ✓", "~0.00%"],
                    "Isolated AUC":       ["0.831", "0.770", "0.780", "0.778"],
                    "Chained AUC":        ["0.835", "0.768", "0.781", "0.778"],
                })
                st.dataframe(chaining_df, use_container_width=True, hide_index=True)

                st.markdown("""
                <div class="insight-card" style="margin-top:14px">
                  <p>
                    <strong style="color:var(--teal-600)">Key Finding —</strong>
                    Chaining improves most on the BRFSS dataset (+1.40% accuracy), where metabolic syndrome
                    comorbidity burden is highest. This supports the hypothesis that upstream Diabetes and CKD
                    risk adds signal specifically for patients with multiple concurrent risk factors.
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
                    Dataset biases: PIMA Indians dataset skews female South Asian; BRFSS is self-reported.
                    CKD model uses reduced features — full lab panel needed for clinical CKD diagnosis.
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
                sim_pa = st.select_slider("Physical Activity", options=["Low", "Moderate", "High"], value=phys_act, key="sim_pa")
                sim_smoking = st.selectbox("Smoking", ["Never", "Former", "Current"], index=["Never", "Former", "Current"].index(smoking), key="sim_smoking")
                
            with sim_col2:
                # Re-run prediction with simulated inputs
                sim_data = encode_inputs(
                    age, sex, sim_bmi, sim_sbp, diastolic_bp,
                    sim_glucose, cholesterol, sim_smoking, alcohol, sim_pa, fam_hist,
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

