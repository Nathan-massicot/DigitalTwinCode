"""Streamlit page: pmsampsize — minimum sample size for a clinical prediction model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from engine.pmsampsize import (
    pmsampsize_binary,
    pmsampsize_continuous,
    pmsampsize_survival,
    max_r2_cs_binary,
    r2cs_from_cstat,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _criteria_table(result, has_events: bool) -> pd.DataFrame:
    rows = []
    for name, c in result.criteria.items():
        row = {
            "Criterion": name,
            "Required N": c["n"],
            "Rationale": c["explanation"],
        }
        if has_events:
            row["Events (E)"] = c["events"]
            row["EPP (E/P)"] = round(c["epp"], 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["Criterion", "Required N"]
    if has_events:
        cols += ["Events (E)", "EPP (E/P)"]
    cols += ["Rationale"]
    return df[cols]


def _final_metrics(result, has_events: bool):
    cols = st.columns(4 if has_events else 2)
    with cols[0]:
        st.metric("Minimum N", f"{result.final_n:,}")
    with cols[1]:
        st.metric("Parameters (P)", result.parameters)
    if has_events:
        with cols[2]:
            st.metric("Expected events (E)", f"{result.final_events:,}")
        with cols[3]:
            st.metric("EPP (E/P)", f"{result.final_epp:.2f}")


# ---------------------------------------------------------------------------
# Outcome-specific UIs
# ---------------------------------------------------------------------------

def _render_continuous():
    st.markdown(
        "Sample size for a **linear regression** prediction model "
        "(Riley *et al.*, Stat Med 2019, Part I)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        parameters = st.number_input("Candidate parameters (P)", min_value=1, value=25, step=1, key="cont_p")
        rsquared = st.number_input(
            "Anticipated R² (adjusted)", min_value=0.001, max_value=0.999,
            value=0.20, step=0.01, format="%.3f", key="cont_r2",
        )
    with c2:
        intercept = st.number_input("Anticipated outcome mean", value=10.0, step=0.5, key="cont_mu")
        sd = st.number_input("Anticipated outcome SD", min_value=0.001, value=2.0, step=0.1, key="cont_sd")
    with c3:
        mape = st.number_input(
            "Margin of error (MAPE)", min_value=0.005, max_value=0.5,
            value=0.05, step=0.005, format="%.3f", key="cont_mape",
        )
        shrinkage = st.number_input(
            "Targeted shrinkage S", min_value=0.5, max_value=0.999,
            value=0.9, step=0.01, format="%.3f", key="cont_S",
        )

    try:
        result = pmsampsize_continuous(
            parameters=int(parameters),
            rsquared=float(rsquared),
            intercept=float(intercept),
            sd=float(sd),
            mape=float(mape),
            shrinkage=float(shrinkage),
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return

    st.markdown("---")
    st.subheader("Result")
    _final_metrics(result, has_events=False)

    st.dataframe(_criteria_table(result, has_events=False), use_container_width=True, hide_index=True)

    st.caption(
        "The minimum N is the **maximum** across the four criteria — it satisfies all of them simultaneously."
    )


def _render_binary():
    st.markdown(
        "Sample size for a **binary outcome** prediction model "
        "(Riley *et al.*, Stat Med 2019, Part II)."
    )

    c1, c2 = st.columns(2)
    with c1:
        parameters = st.number_input("Candidate parameters (P)", min_value=1, value=24, step=1, key="bin_p")
        prevalence = st.number_input(
            "Outcome prevalence φ", min_value=0.001, max_value=0.999,
            value=0.174, step=0.01, format="%.3f", key="bin_phi",
        )
    with c2:
        shrinkage = st.number_input(
            "Targeted shrinkage S", min_value=0.5, max_value=0.999,
            value=0.9, step=0.01, format="%.3f", key="bin_S",
        )
        mape_prev = st.number_input(
            "Margin of error on prevalence", min_value=0.005, max_value=0.5,
            value=0.05, step=0.005, format="%.3f", key="bin_mape",
        )

    st.markdown("**Anticipated discrimination** — provide *one* of the following")
    mode = st.radio(
        "Specify model fit by",
        options=["Cox-Snell R²", "C-statistic (AUC)"],
        horizontal=True, key="bin_mode",
    )

    rsquared_cs = c_statistic = None
    max_r2 = max_r2_cs_binary(float(prevalence))

    if mode == "Cox-Snell R²":
        rsquared_cs = st.number_input(
            f"Anticipated Cox-Snell R²  (max for this prevalence ≈ {max_r2:.3f})",
            min_value=0.001, max_value=float(max_r2) - 1e-4,
            value=min(0.05, float(max_r2) * 0.5), step=0.005, format="%.4f",
            key="bin_r2cs",
        )
    else:
        c_statistic = st.number_input(
            "Anticipated C-statistic (AUC)",
            min_value=0.501, max_value=0.999, value=0.75, step=0.01, format="%.3f",
            key="bin_cstat",
        )
        try:
            implied = r2cs_from_cstat(float(c_statistic), float(prevalence))
            st.caption(f"Implied Cox-Snell R² ≈ **{implied:.4f}** (max ≈ {max_r2:.3f}).")
        except Exception as exc:
            st.warning(f"Could not derive R² from C-statistic: {exc}")

    try:
        result = pmsampsize_binary(
            parameters=int(parameters),
            prevalence=float(prevalence),
            rsquared_cs=float(rsquared_cs) if rsquared_cs is not None else None,
            c_statistic=float(c_statistic) if c_statistic is not None else None,
            shrinkage=float(shrinkage),
            mape_prevalence=float(mape_prev),
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return

    st.markdown("---")
    st.subheader("Result")
    _final_metrics(result, has_events=True)

    st.dataframe(_criteria_table(result, has_events=True), use_container_width=True, hide_index=True)

    st.caption(
        "The minimum N is the **maximum** across the three criteria. "
        "EPP = Events Per candidate Parameter; the legacy '10 EPP rule' is "
        "now considered insufficient (see Riley 2020, BMJ)."
    )


def _render_survival():
    st.markdown(
        "Sample size for a **time-to-event** prediction model "
        "(Riley *et al.*, Stat Med 2019, Part II)."
    )

    c1, c2 = st.columns(2)
    with c1:
        parameters = st.number_input("Candidate parameters (P)", min_value=1, value=30, step=1, key="surv_p")
        rate = st.number_input(
            "Overall event rate (per person-time unit)", min_value=1e-6,
            value=0.065, step=0.005, format="%.4f", key="surv_rate",
        )
        timepoint = st.number_input("Time point of interest", min_value=0.01, value=2.0, step=0.5, key="surv_tp")
    with c2:
        meanfup = st.number_input("Mean follow-up", min_value=0.01, value=2.07, step=0.1, key="surv_fup")
        rsquared_cs = st.number_input(
            "Anticipated Cox-Snell R²", min_value=0.001, max_value=0.999,
            value=0.051, step=0.005, format="%.4f", key="surv_r2cs",
        )
        shrinkage = st.number_input(
            "Targeted shrinkage S", min_value=0.5, max_value=0.999,
            value=0.9, step=0.01, format="%.3f", key="surv_S",
        )

    try:
        result = pmsampsize_survival(
            parameters=int(parameters),
            rate=float(rate),
            timepoint=float(timepoint),
            meanfup=float(meanfup),
            rsquared_cs=float(rsquared_cs),
            shrinkage=float(shrinkage),
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return

    st.markdown("---")
    st.subheader("Result")
    _final_metrics(result, has_events=True)

    st.dataframe(_criteria_table(result, has_events=True), use_container_width=True, hide_index=True)

    st.caption(
        f"Implied event proportion at mean follow-up = "
        f"{result.inputs['proportion_event']:.3f}; "
        f"max Cox-Snell R² ≈ {result.inputs['max_r2_cs']:.3f}."
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_pmsampsize():
    st.title("pmsampsize — Minimum sample size for a prediction model")
    st.markdown(
        "Python port of the **pmsampsize** Stata/R package "
        "(Ensor, Martin, Riley). Computes the minimum N satisfying the "
        "Riley *et al.* criteria for developing a multivariable prediction model."
    )

    outcome = st.radio(
        "Outcome type",
        options=["Continuous", "Binary", "Survival (time-to-event)"],
        horizontal=True,
        key="pms_outcome",
    )

    st.markdown("---")

    if outcome == "Continuous":
        _render_continuous()
    elif outcome == "Binary":
        _render_binary()
    else:
        _render_survival()

    st.markdown("---")
    with st.expander("Methodological notes"):
        st.markdown(
            "- **Criterion 1** — small overall shrinkage (S = 0.9 → ≤10 % shrinkage of predictor effects).\n"
            "- **Criterion 2** — small absolute difference (≤0.05) between the model's apparent "
            "and optimism-adjusted (Cox-Snell) R².\n"
            "- **Criterion 3** — precise estimation of the overall outcome (proportion / rate).\n"
            "- **Criterion 4 (continuous only)** — precise estimation of the residual SD and intercept.\n"
            "\n"
            "The reported **minimum N** is the maximum across all criteria; this guarantees they are "
            "all satisfied simultaneously. For binary/survival outcomes, the **expected events (E)** "
            "and **EPP (E/P)** are reported alongside N."
        )
        st.markdown(
            "**References** — Riley RD, Snell KIE, Ensor J, et al. Stat Med 2019; "
            "Riley RD, Ensor J, Snell KIE, et al. BMJ 2020;368:m441."
        )
