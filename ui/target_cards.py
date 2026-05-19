import streamlit as st


def _prob_color(prob: float) -> str:
    if prob > 0.6:
        return "#16a34a"
    elif prob > 0.4:
        return "#ca8a04"
    return "#dc2626"


def render_target_cards(predictions: dict):
    """Render the 4 target prediction cards (mock prototype)."""
    fim_d = predictions["fim_discharge"]
    fim_g = predictions["fim_gain"]
    m_prob = predictions["meaningful_prob"]
    rh_prob = predictions["return_home_prob"]

    st.markdown(
        '<style>div[data-testid="stMetric"] { margin: 10px; }</style>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4, gap="medium")

    with cols[0]:
        st.metric(
            label="FIM Total at Discharge",
            value=f"{fim_d:.0f}",
            delta=f"+{fim_g:.0f} from entry" if fim_g >= 0 else f"{fim_g:.0f} from entry",
        )

    with cols[1]:
        st.metric(
            label="FIM Gain",
            value=f"{fim_g:+.0f}",
            delta="Good" if fim_g >= 15 else "Low",
            delta_color="normal" if fim_g >= 15 else "inverse",
        )

    with cols[2]:
        color = _prob_color(m_prob)
        st.markdown(
            f"""
            <div style="background:#f8fafc; margin:10px; border-radius:12px; padding:16px; border:1px solid #e2e8f0;">
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">Meaningful Improvement</div>
                <div style="font-size:0.85rem; color:#64748b;">(FIM gain &ge; 15)</div>
                <div style="font-size:2rem; font-weight:700; color:{color};">{m_prob*100:.0f}%</div>
                <div style="font-size:0.9rem;">{'Yes' if m_prob > 0.5 else 'No'}
                    <span style="color:{color};">&#9679;</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        color = _prob_color(rh_prob)
        st.markdown(
            f"""
            <div style="background:#f8fafc; margin:10px; border-radius:12px; padding:16px; border:1px solid #e2e8f0;">
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">Return Home</div>
                <div style="font-size:2rem; font-weight:700; color:{color};">{rh_prob*100:.0f}%</div>
                <div style="font-size:0.9rem;">{'Yes' if rh_prob > 0.5 else 'No'}
                    <span style="color:{color};">&#9679;</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


_CARD_CSS = """
<style>
.dt-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 16px;
    height: 170px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    margin: 6px 0;
}
.dt-card .dt-title {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 600;
}
.dt-card .dt-subtitle {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
    min-height: 1rem;
}
.dt-card .dt-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.1;
}
.dt-card .dt-footer {
    font-size: 0.9rem;
    color: #475569;
}
</style>
"""


def _card_html(title: str, value: str, footer: str,
               subtitle: str = "", value_color: str = "#1e293b",
               footer_color: str = "#475569") -> str:
    return f"""
    <div class="dt-card">
        <div>
            <div class="dt-title">{title}</div>
            <div class="dt-subtitle">{subtitle}</div>
        </div>
        <div class="dt-value" style="color:{value_color};">{value}</div>
        <div class="dt-footer" style="color:{footer_color};">{footer}</div>
    </div>
    """


def render_real_target_cards(predictions: dict):
    """Render 6 outcome prediction cards for the real model."""
    from models.config import (
        REGRESSION_TARGETS,
        CLASSIFICATION_TARGET,
        MEANINGFUL_THRESHOLD,
    )

    fim_total = predictions.get(REGRESSION_TARGETS[0], 0)
    fim_motor = predictions.get(REGRESSION_TARGETS[1], 0)
    fim_cognitive = predictions.get(REGRESSION_TARGETS[2], 0)
    fim_gain = predictions.get("fim_gain", 0)
    m_prob = predictions.get("meaningful_prob", float(predictions.get("meaningful_improvement", 0)))
    rh_prob = predictions.get("return_home_prob", float(predictions.get(CLASSIFICATION_TARGET, 0)))

    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    gain_color = "#16a34a" if fim_gain >= MEANINGFUL_THRESHOLD else "#dc2626"
    gain_label = "Good" if fim_gain >= MEANINGFUL_THRESHOLD else "Low"

    # Row 1: FIM scores
    cols = st.columns(3, gap="medium")
    with cols[0]:
        st.markdown(
            _card_html(
                title="FIM Total at Discharge",
                subtitle="Functional Independence Measure",
                value=f"{fim_total:.0f}",
                footer=f"{fim_gain:+.0f} from entry",
                footer_color=gain_color,
            ),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            _card_html(
                title="FIM Motor at Discharge",
                subtitle="Mobility & self-care subscale",
                value=f"{fim_motor:.0f}",
                footer="out of 91",
            ),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            _card_html(
                title="FIM Cognitive at Discharge",
                subtitle="Communication & cognition subscale",
                value=f"{fim_cognitive:.0f}",
                footer="out of 35",
            ),
            unsafe_allow_html=True,
        )

    # Row 2: derived + classification
    cols2 = st.columns(3, gap="medium")
    with cols2[0]:
        st.markdown(
            _card_html(
                title="FIM Gain",
                subtitle="Discharge − Entry",
                value=f"{fim_gain:+.0f}",
                footer=gain_label,
                value_color=gain_color,
                footer_color=gain_color,
            ),
            unsafe_allow_html=True,
        )
    with cols2[1]:
        m_color = _prob_color(m_prob)
        m_label = "Yes" if m_prob > 0.5 else "No"
        st.markdown(
            _card_html(
                title="Meaningful Improvement",
                subtitle=f"FIM gain ≥ {MEANINGFUL_THRESHOLD}",
                value=f"{m_prob*100:.0f}%",
                footer=f"{m_label} ●",
                value_color=m_color,
                footer_color=m_color,
            ),
            unsafe_allow_html=True,
        )
    with cols2[2]:
        rh_color = _prob_color(rh_prob)
        rh_label = "Yes" if rh_prob > 0.5 else "No"
        st.markdown(
            _card_html(
                title="Return Home",
                subtitle="Discharge destination",
                value=f"{rh_prob*100:.0f}%",
                footer=f"{rh_label} ●",
                value_color=rh_color,
                footer_color=rh_color,
            ),
            unsafe_allow_html=True,
        )
