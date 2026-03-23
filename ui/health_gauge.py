import plotly.graph_objects as go
import streamlit as st


def render_health_gauge(health: dict):
    """Render the red-yellow-green gauge with boxplot overlay."""
    score = health["score"]

    fig = go.Figure()

    # Gradient gauge background using a series of thin bars
    n_segments = 200
    for i in range(n_segments):
        x_val = i / n_segments * 100
        ratio = x_val / 100
        if ratio < 0.33:
            t = ratio / 0.33
            r = int(220 - t * 40)
            g = int(60 + t * 140)
            b = 60
        elif ratio < 0.66:
            t = (ratio - 0.33) / 0.33
            r = int(180 - t * 120)
            g = int(200 - t * 10)
            b = int(60 + t * 20)
        else:
            t = (ratio - 0.66) / 0.34
            r = int(60 - t * 10)
            g = int(190 - t * 30)
            b = int(80 - t * 20)
        color = f"rgb({r},{g},{b})"
        fig.add_shape(
            type="rect",
            x0=x_val, x1=x_val + 100 / n_segments + 0.1,
            y0=-0.4, y1=0.4,
            fillcolor=color,
            line_width=0,
            layer="below",
        )

    # Boxplot overlay
    fig.add_trace(go.Box(
        x=[health["whisker_low"], health["q1"], health["median"],
           health["q3"], health["whisker_high"]],
        q1=[health["q1"]],
        median=[health["median"]],
        q3=[health["q3"]],
        lowerfence=[health["whisker_low"]],
        upperfence=[health["whisker_high"]],
        orientation="h",
        marker_color="rgba(30,30,30,0.8)",
        line_color="rgba(30,30,30,0.9)",
        fillcolor="rgba(255,255,255,0.5)",
        width=0.5,
        name="Confidence",
        hoverinfo="x",
    ))

    # Predicted value marker
    fig.add_trace(go.Scatter(
        x=[score],
        y=[0],
        mode="markers",
        marker=dict(size=16, color="black", symbol="diamond"),
        name=f"Predicted: {score:.0f}",
        hovertemplate=f"Score: {score:.1f}<extra></extra>",
    ))

    fig.update_layout(
        height=140,
        margin=dict(l=20, r=20, t=60, b=60,),
        xaxis=dict(range=[0, 100], title="Health Score", dtick=10),
        yaxis=dict(visible=False, range=[-0.8, 0.8]),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:20px;">
                <div style="font-size:2.2rem; font-weight:700; color:#1e293b;">{score:.0f}<span style="font-size:1rem; color:#64748b;">/100</span></div>
                <div style="font-size:0.9rem; color:#64748b;">Confidence: &pm;{health['confidence']:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
