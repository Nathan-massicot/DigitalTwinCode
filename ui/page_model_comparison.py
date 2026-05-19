"""Streamlit page: Model Comparison — regression tables, confusion matrices, scatter plots, best model."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.config import (
    FEATURE_SETS,
    ALGORITHMS,
    REGRESSION_TARGETS,
    CLASSIFICATION_TARGET,
    TARGET_LABELS,
    ALGO_LABELS,
    ALGO_COLORS,
)
from models.results import get_all_results, load_real_data


# ---------------------------------------------------------------------------
# Section 1 — Regression summary table
# ---------------------------------------------------------------------------

def _build_regression_table(results: dict) -> pd.DataFrame:
    """Build a pivot table: rows = targets, columns = (feature_set, algo), cells = MAE / R2."""
    rows = []
    for target in REGRESSION_TARGETS:
        label = TARGET_LABELS[target]
        for fs_name in FEATURE_SETS:
            for algo in ALGORITHMS:
                key = (fs_name, algo, target)
                r = results.get(key)
                if r is None:
                    continue
                rows.append({
                    "Target": label,
                    "Features": fs_name,
                    "Algorithm": ALGO_LABELS[algo],
                    "MAE": r["MAE"],
                    "R2": r["R2"],
                })
    return pd.DataFrame(rows)


def _render_regression_heatmap(df_table: pd.DataFrame):
    """Render coloured regression metrics table."""
    st.subheader("Regression Performance (LOOCV)")

    for metric, ascending, fmt in [("MAE", True, "{:.2f}"), ("R2", False, "{:.3f}")]:
        st.markdown(f"**{metric}**")
        pivot = df_table.pivot_table(index="Target", columns=["Features", "Algorithm"], values=metric)
        col_order = []
        for fs in FEATURE_SETS:
            for algo in ALGORITHMS:
                col_key = (fs, ALGO_LABELS[algo])
                if col_key in pivot.columns:
                    col_order.append(col_key)
        pivot = pivot[col_order]

        if ascending:
            styled = pivot.style.background_gradient(cmap="RdYlGn_r", axis=None).format(fmt)
        else:
            styled = pivot.style.background_gradient(cmap="RdYlGn", axis=None).format(fmt)

        st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 2 — Confusion matrices (3×3 grid)
# ---------------------------------------------------------------------------

def _render_confusion_grid(results: dict, target_key_template: str, section_title: str):
    """Render a 3×3 grid of confusion matrices for the given target."""
    st.subheader(section_title)

    fs_names = list(FEATURE_SETS.keys())

    for fs_name in fs_names:
        st.markdown(f"**{fs_name}**")
        cols = st.columns(3)
        for i, algo in enumerate(ALGORITHMS):
            key = (fs_name, algo, target_key_template)
            r = results.get(key)
            if r is None:
                with cols[i]:
                    st.warning(f"No results for {ALGO_LABELS[algo]}")
                continue

            cm = r["Confusion_Matrix"]
            with cols[i]:
                # Heatmap
                labels = ["No (0)", "Yes (1)"]
                text = [[str(cm[row][col]) for col in range(2)] for row in range(2)]

                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 18},
                    colorscale="Blues",
                    showscale=False,
                ))
                fig.update_layout(
                    title=dict(text=ALGO_LABELS[algo], font=dict(
                        size=14, color=ALGO_COLORS[algo]
                    )),
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=280,
                    margin=dict(l=60, r=20, t=40, b=60),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"cm_{section_title}_{fs_name}_{algo}")

                st.caption(
                    f"Acc={r['Accuracy']:.3f}  Sens={r['Sensitivity']:.3f}  "
                    f"Spec={r['Specificity']:.3f}  F1={r['F1']:.3f}"
                )


# ---------------------------------------------------------------------------
# Section 3 — Scatter plots (Predicted vs Actual)
# ---------------------------------------------------------------------------

def _render_scatter_plots(results: dict, df: pd.DataFrame):
    st.subheader("Predicted vs Actual (Regression)")

    target_sel = st.selectbox(
        "Target", REGRESSION_TARGETS,
        format_func=lambda t: TARGET_LABELS[t],
        key="scatter_target",
    )

    col_fs, col_algo = st.columns(2)
    with col_fs:
        fs_sel = st.selectbox("Feature set", list(FEATURE_SETS.keys()), key="scatter_fs")
    with col_algo:
        algo_sel = st.selectbox("Algorithm", ALGORITHMS, format_func=lambda a: ALGO_LABELS[a], key="scatter_algo")

    key = (fs_sel, algo_sel, target_sel)
    r = results.get(key)
    if r is None:
        st.warning("No results for this combination.")
        return

    y_true = r["y_true"]
    y_pred = r["y_pred"]
    patient_ids = df["patient_id"].values

    min_val = min(y_true.min(), y_pred.min()) - 5
    max_val = max(y_true.max(), y_pred.max()) + 5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(size=10, color=ALGO_COLORS[algo_sel], line=dict(width=1, color="white")),
        text=patient_ids,
        hovertemplate="Patient: %{text}<br>Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(dash="dash", color="grey"),
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=450,
        margin=dict(l=60, r=20, t=30, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"MAE={r['MAE']}  R2={r['R2']}")


# ---------------------------------------------------------------------------
# Section 4 — Best model summary
# ---------------------------------------------------------------------------

def _render_best_model_summary(results: dict):
    st.subheader("Best Model per Target")

    # Regression: best by lowest MAE
    for target in REGRESSION_TARGETS:
        best_key, best_mae = None, float("inf")
        for fs_name in FEATURE_SETS:
            for algo in ALGORITHMS:
                key = (fs_name, algo, target)
                r = results.get(key)
                if r and r["MAE"] < best_mae:
                    best_mae = r["MAE"]
                    best_key = key
        if best_key:
            r = results[best_key]
            st.success(
                f"**{TARGET_LABELS[target]}**: {ALGO_LABELS[best_key[1]]} + {best_key[0]}  "
                f"— MAE={r['MAE']}, R2={r['R2']}"
            )

    # Classification: best by F1
    for target in [CLASSIFICATION_TARGET, "DERIVED__meaningful_improvement"]:
        best_key, best_f1 = None, -1.0
        for fs_name in FEATURE_SETS:
            for algo in ALGORITHMS:
                key = (fs_name, algo, target)
                r = results.get(key)
                if r and r["F1"] > best_f1:
                    best_f1 = r["F1"]
                    best_key = key
        if best_key:
            r = results[best_key]
            st.success(
                f"**{TARGET_LABELS[target]}**: {ALGO_LABELS[best_key[1]]} + {best_key[0]}  "
                f"— F1={r['F1']}, Acc={r['Accuracy']}, Sens={r['Sensitivity']}, Spec={r['Specificity']}"
            )



# ---------------------------------------------------------------------------
# Section 5 — Health Status composite score: design rationale
# ---------------------------------------------------------------------------

def _render_health_score_rationale():
    st.subheader("Prediction of Health Status — Design Rationale")

    st.markdown(
        "The **Health Status** score on the Digital Twin page is a single 0–100 "
        "number that rolls up the five model outputs. It exists as a *triage signal* "
        "— a clinician can glance at one gauge to judge \"is this patient predicted "
        "to do well?\" "
    )

    st.markdown("**Composite formula**")
    st.latex(
        r"\text{Score} = 0.30\cdot\widetilde{FIM}_{total} + 0.20\cdot\widetilde{FIM}_{motor}"
        r" + 0.15\cdot\widetilde{FIM}_{cog} + 0.20\cdot\widetilde{FIM}_{gain}"
        r" + 0.15\cdot P(\text{return home})"
    )
    st.caption(
        "Each FIM component is min-max rescaled to 0–100 over its clinical range "
        "(Total 18–126, Motor 13–91, Cognitive 5–35, Gain −20 to +80); "
        "Return Home enters as a probability in 0–1, scaled to 0–100."
    )

    st.markdown("**Why these components and these weights**")
    st.markdown(
        "- **FIM Total (30%)** — the primary discharge outcome; highest weight.\n"
        "- **FIM Motor (20%) + Cognitive (15%)** — mathematically Motor + Cognitive ≈ "
        "Total, so this looks redundant, but each comes from an *independently trained* "
        "model (different LOOCV-best algorithm, different residuals). Including them "
        "acts as a mini-ensemble and lets motor carry more weight than cognitive, "
        "which matches clinical reality in stroke rehab: motor deficits dominate and "
        "have more room to change; cognition often sits near ceiling.\n"
        "- **FIM Gain (20%)** — progress matters as much as final state. Two patients "
        "at FIM 90 look equal until you see one entered at 85 and the other at 40.\n"
        "- **Return Home (15%)** — a social/environmental outcome the FIM alone can't "
        "capture."
    )


    st.markdown("**The confidence box on the gauge**")
    st.latex(
        r"q_{1,3} = \text{score} \pm 0.8\cdot\sigma_{\text{components}}\qquad"
        r"\text{whiskers} = \text{score} \pm 1.5\cdot\sigma_{\text{components}}"
    )
    st.markdown(
        "This is **not** a statistical prediction interval, it is a **coherence "
        "indicator**. When the 5 normalized components agree, σ is small → tight box "
        "→ \"the evidence is consistent.\" When they disagree (e.g. high FIM Total "
        "but low Return Home probability), σ grows → wide box → \"something about "
        "this case is contradictory, read the detail cards.\""
    )

    st.markdown("**Honest tradeoffs**")
    st.markdown(
        "1. **Weights are clinical judgment, not data-fit.** "
        "2. **FIM Gain range (−20 to +80) is empirical.** If the cohort widens, the "
        "normalization drifts.\n"
        "3. **The confidence box is a heuristic, not a true interval.** It is based "
        "on the spread of the 5 components, not on the actual residuals of the composite score. It is meant to be a *relative* indicator of agreement, not an absolute measure of uncertainty."
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_model_comparison():
    st.title("Model Comparison — LOOCV on 20 Real Patients")
    st.markdown(
        "Comparison of **3 algorithms** (k-NN, Ridge/Logistic, Random Forest) "
        "across **3 feature subsets** (Small 5, Medium 12, Full 21). "
        "Validation: Leave-One-Out Cross-Validation."
    )

    with st.spinner("Running LOOCV pipeline (first load only)..."):
        results = get_all_results()
    df = load_real_data()

    st.markdown("---")

    # Section 1: Regression table
    df_table = _build_regression_table(results)
    _render_regression_heatmap(df_table)

    st.markdown("---")

    # Section 2: Confusion matrices
    tab_rh, tab_mi = st.tabs(["Return Home", "Meaningful Improvement"])
    with tab_rh:
        _render_confusion_grid(results, CLASSIFICATION_TARGET, "Return Home — Confusion Matrices")
    with tab_mi:
        _render_confusion_grid(results, "DERIVED__meaningful_improvement", "Meaningful Improvement (derived) — Confusion Matrices")

    st.markdown("---")

    # Section 3: Scatter plots
    _render_scatter_plots(results, df)

    st.markdown("---")

    # Section 4: Best model
    _render_best_model_summary(results)

    st.markdown("---")

    # Section 5: Health Status composite design rationale
    _render_health_score_rationale()
