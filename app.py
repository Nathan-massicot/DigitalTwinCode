import streamlit as st

from data.mock_patients import generate_cohort
from engine.predictor import predict
from engine.scoring import compute_health_score
from ui.sliders import render_sliders
from ui.health_gauge import render_health_gauge
from ui.target_cards import render_target_cards

st.set_page_config(
    page_title="Digital Twin — Stroke Rehab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean medical style
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    h1 { color: #1e293b; font-weight: 700; }
    h2 { color: #334155; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem; }
    .stRadio > div { flex-direction: row; gap: 1rem; }
    .stRadio > div > label {
        background: #f1f5f9; padding: 0.4rem 1rem; border-radius: 8px;
        border: 1px solid #cbd5e1; font-weight: 500;
    }
    .stRadio > div > label[data-checked="true"] {
        background: #3b82f6; color: white; border-color: #3b82f6;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc; border-radius: 12px; padding: 16px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cohort():
    return generate_cohort(n=200, seed=42)


cohort = load_cohort()

# --- Header ---
st.title("Digital Twin — Stroke Rehabilitation Prototype")

horizon = st.radio(
    "Prediction Horizon",
    options=["T0 — Rehab Entry", "T1 — Day 7-14", "OUTCOME — Discharge"],
    index=2,
    horizontal=True,
)
horizon_key = horizon.split(" —")[0].strip()

st.markdown("---")

# --- Parameters ---
st.header("Patient Parameters (Top 20 Features by Importance)")
patient_values = render_sliders()

st.markdown("---")

# --- Predictions ---
predictions = predict(patient_values, cohort, k=20, horizon=horizon_key)
health = compute_health_score(predictions)

# --- Health Status ---
st.header("Prediction of Health Status")
render_health_gauge(health)

st.markdown("---")

# --- Target Values ---
st.header("Prediction of Target Values")
render_target_cards(predictions)
