import streamlit as st


def _prob_color(prob: float) -> str:
    if prob > 0.6:
        return "#16a34a"
    elif prob > 0.4:
        return "#ca8a04"
    return "#dc2626"


def render_target_cards(predictions: dict):
    """Render the 4 target prediction cards."""
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
