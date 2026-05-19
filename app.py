import streamlit as st

st.set_page_config(
    page_title="Digital Twin — Stroke Rehab",
    layout="wide",
    initial_sidebar_state="expanded",
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

# --- Navigation ---
page = st.sidebar.radio("Navigation", [
    "Digital Twin",
    "Digital Twin Prototype (mock)",
    "Model Comparison",
    "Sample Size (pmsampsize)",
])

if page == "Digital Twin":
    from data.real_features import REAL_FEATURE_DEFS
    from engine.real_predictor import predict_patient
    from engine.scoring import compute_real_health_score
    from models.results import get_trained_best_models
    from ui.sliders import render_sliders
    from ui.health_gauge import render_health_gauge
    from ui.target_cards import render_real_target_cards

    st.title("Digital Twin — Stroke Rehabilitation")

    with st.spinner("Loading models..."):
        trained_models = get_trained_best_models()


    # --- Parameters ---
    st.header("Patient Parameters (21 Features)")
    patient_values = render_sliders(feature_defs=REAL_FEATURE_DEFS, key_prefix="real")

    st.markdown("---")

    # --- Predictions ---
    predictions = predict_patient(patient_values, trained_models)

    health = compute_real_health_score(predictions)
    st.header("Prediction of Health Status")
    st.markdown(
        "Composite score (0–100) combining the five predicted outcomes, "
        "each rescaled to 0–100 and weighted by clinical relevance: "
        "**FIM Total (30%)**, **FIM Motor (20%)**, **FIM Cognitive (15%)**, "
        "**FIM Gain (20%)**, and **Return Home probability (15%)**. "
        "The boxplot shows the confidence interval estimated from the spread of the components."
    )
    render_health_gauge(health)

    st.markdown("---")

    st.header("Predicted Outcomes")
    render_real_target_cards(predictions)

elif page == "Digital Twin Prototype (mock)":
    from data.mock_patients import generate_cohort
    from engine.predictor import predict
    from engine.scoring import compute_health_score
    from ui.sliders import render_sliders
    from ui.health_gauge import render_health_gauge
    from ui.target_cards import render_target_cards

    @st.cache_data
    def load_cohort():
        return generate_cohort(n=200, seed=42)

    cohort = load_cohort()

    st.title("Digital Twin — Stroke Rehabilitation Prototype")

    horizon = st.radio(
        "Prediction Horizon",
        options=["T0 — Rehab Entry", "T1 — Day 7-14", "OUTCOME — Discharge"],
        index=2,
        horizontal=True,
    )
    horizon_key = horizon.split(" —")[0].strip()

    st.header("Patient Parameters (Top 20 Features by Importance)")
    patient_values = render_sliders()

    st.markdown("---")

    predictions = predict(patient_values, cohort, k=20, horizon=horizon_key)
    health = compute_health_score(predictions)

    st.header("Prediction of Health Status")
    render_health_gauge(health)

    st.markdown("---")

    st.header("Prediction of Target Values")
    render_target_cards(predictions)

elif page == "Model Comparison":
    from ui.page_model_comparison import render_model_comparison
    render_model_comparison()

elif page == "Sample Size (pmsampsize)":
    from ui.page_pmsampsize import render_pmsampsize
    render_pmsampsize()
