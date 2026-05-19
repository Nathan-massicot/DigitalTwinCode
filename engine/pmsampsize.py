"""Port of the pmsampsize R/Stata package (Riley et al. 2018, 2020).

Computes the minimum sample size required for developing a clinical prediction
model with continuous, binary, or time-to-event outcomes. Returns the maximum
sample size satisfying each of three (continuous) or four (binary, survival)
criteria proposed by Riley et al.

References:
    Riley RD, et al. Minimum sample size for developing a multivariable
    prediction model: Part I — Continuous outcomes. Stat Med. 2019.
    Riley RD, et al. Minimum sample size for developing a multivariable
    prediction model: PART II — binary and time-to-event outcomes. Stat Med. 2019.
    Riley RD, et al. Calculating the sample size required for developing
    a clinical prediction model. BMJ. 2020;368:m441.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_975 = 1.959964  # 97.5% normal quantile


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def r2cs_from_cstat(c_stat: float, prevalence: float, n_sim: int = 500_000, seed: int = 42) -> float:
    """Approximate Cox-Snell R² from an anticipated C-statistic + prevalence.

    Direct port of pmsampsize's `approximate_R2`: simulates a case-control
    sample with non-events ~ N(0, 1) and events ~ N(μ, 1), where
    μ = √2 · Φ⁻¹(C). The intercept is solved so that the average predicted
    probability equals the target prevalence; the Cox-Snell R² is then read
    off the resulting log-likelihood.
    """
    if not 0.5 < c_stat < 1.0:
        raise ValueError("C-statistic must be in (0.5, 1.0).")
    if not 0.0 < prevalence < 1.0:
        raise ValueError("Prevalence must be in (0, 1).")

    import numpy as np
    from statistics import NormalDist

    rng = np.random.default_rng(seed)
    mu = math.sqrt(2) * NormalDist().inv_cdf(c_stat)

    L = rng.normal(loc=0.0, scale=1.0, size=n_sim)
    Lp = rng.normal(loc=mu, scale=1.0, size=n_sim)
    Lall = np.concatenate([L, Lp])

    # Solve intercept: mean(expit(α + Lall)) = prevalence  (bisection)
    lo, hi = -30.0, 30.0
    for _ in range(80):
        a = 0.5 * (lo + hi)
        m = float(np.mean(1.0 / (1.0 + np.exp(-(a + Lall)))))
        if m < prevalence:
            lo = a
        else:
            hi = a
    alpha = 0.5 * (lo + hi)

    pi = 1.0 / (1.0 + np.exp(-(alpha + Lall)))
    Y = (rng.uniform(size=Lall.size) < pi).astype(float)

    pi_clip = np.clip(pi, 1e-15, 1 - 1e-15)
    ll_model = float(np.mean(Y * np.log(pi_clip) + (1 - Y) * np.log(1 - pi_clip)))
    p_obs = float(np.mean(Y))
    p_obs = min(max(p_obs, 1e-15), 1 - 1e-15)
    ll_null = p_obs * math.log(p_obs) + (1 - p_obs) * math.log(1 - p_obs)

    return 1.0 - math.exp(2.0 * (ll_null - ll_model))


def max_r2_cs_binary(prevalence: float) -> float:
    """Maximum achievable Cox-Snell R² for a binary outcome (Nagelkerke 1991)."""
    ll_null = prevalence * math.log(prevalence) + (1 - prevalence) * math.log(1 - prevalence)
    return 1.0 - math.exp(2.0 * ll_null)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SampleSizeResult:
    outcome_type: str
    parameters: int
    criteria: dict          # {label: {"n": ..., "epp"/"epv": ..., "shrinkage": ..., "explanation": ...}}
    final_n: int
    final_epp: float | None = None     # events per parameter (binary/survival)
    final_events: int | None = None    # expected events
    inputs: dict | None = None


# ---------------------------------------------------------------------------
# Continuous outcome
# ---------------------------------------------------------------------------

def pmsampsize_continuous(
    parameters: int,
    rsquared: float,
    intercept: float,
    sd: float,
    mape: float = 0.05,
    shrinkage: float = 0.9,
) -> SampleSizeResult:
    """Sample size for a linear regression prediction model (Riley 2019, Part I).

    Parameters
    ----------
    parameters : int
        Number of candidate predictor parameters P.
    rsquared : float
        Anticipated R² (adjusted) of the new model, from prior literature.
    intercept : float
        Anticipated mean of the outcome variable.
    sd : float
        Anticipated SD of the outcome (intrinsic outcome variability).
    mape : float
        Multiplicative margin of error in predictions (default 0.05 → ≤5 %).
    shrinkage : float
        Targeted uniform shrinkage factor S (default 0.9 → ≤10 % shrinkage).
    """
    if not 0 < rsquared < 1:
        raise ValueError("rsquared must be in (0, 1).")
    if not 0 < shrinkage < 1:
        raise ValueError("shrinkage must be in (0, 1).")
    if parameters < 1:
        raise ValueError("parameters must be ≥ 1.")
    if sd <= 0:
        raise ValueError("sd must be > 0.")

    # Criterion 1: small overall shrinkage
    n1 = parameters / ((shrinkage - 1.0) * math.log(1.0 - rsquared / shrinkage))
    n1 = math.ceil(n1)

    # Criterion 2: small absolute difference (≤0.05) between apparent and
    # adjusted R² (using formula from Riley 2019 Part I, Eq. 9)
    # n2 such that S_VH = R²/(R² + 0.05) gives required n
    s_vh = rsquared / (rsquared + 0.05 * (1.0 - rsquared))
    n2 = parameters / ((s_vh - 1.0) * math.log(1.0 - rsquared / s_vh))
    n2 = math.ceil(n2)

    # Criterion 3: precise estimate of σ (residual SD)
    # χ² inversion: n3 = P + 1 + 2*(z/mape)²  (variance scale, Riley 2019 Eq. 10)
    n3 = parameters + 1 + math.ceil(2.0 * (Z_975 / mape) ** 2)

    # Criterion 4: precise estimate of intercept (mean)
    # Half-width of 95% CI on mean ≤ mape × sd  →  z * sd / sqrt(n) ≤ mape*sd
    n4 = math.ceil((Z_975 / mape) ** 2)

    final_n = max(n1, n2, n3, n4)

    criteria = {
        "Criterion 1 — Global shrinkage": {
            "n": n1,
            "explanation": f"Targets uniform shrinkage S = {shrinkage}.",
        },
        "Criterion 2 — Small optimism in R²": {
            "n": n2,
            "explanation": "Targets ≤0.05 difference between apparent and adjusted R².",
        },
        "Criterion 3 — Precision of residual SD": {
            "n": n3,
            "explanation": f"Targets ±{mape:.0%} relative margin of error on σ̂.",
        },
        "Criterion 4 — Precision of intercept": {
            "n": n4,
            "explanation": f"Targets ±{mape:.0%}·SD margin of error on the mean.",
        },
    }

    return SampleSizeResult(
        outcome_type="continuous",
        parameters=parameters,
        criteria=criteria,
        final_n=final_n,
        inputs={
            "rsquared": rsquared,
            "intercept": intercept,
            "sd": sd,
            "mape": mape,
            "shrinkage": shrinkage,
        },
    )


# ---------------------------------------------------------------------------
# Binary outcome
# ---------------------------------------------------------------------------

def pmsampsize_binary(
    parameters: int,
    prevalence: float,
    rsquared_cs: float | None = None,
    c_statistic: float | None = None,
    shrinkage: float = 0.9,
    mape_prevalence: float = 0.05,
) -> SampleSizeResult:
    """Sample size for a binary outcome prediction model (Riley 2019, Part II).

    Provide *either* `rsquared_cs` (anticipated Cox-Snell R²) *or* `c_statistic`
    (anticipated AUC); when only the C-statistic is given, the Cox-Snell R²
    is approximated assuming a normally distributed linear predictor.
    """
    if parameters < 1:
        raise ValueError("parameters must be ≥ 1.")
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be in (0, 1).")
    if not 0.0 < shrinkage < 1.0:
        raise ValueError("shrinkage must be in (0, 1).")

    if rsquared_cs is None and c_statistic is None:
        raise ValueError("Provide rsquared_cs or c_statistic.")
    if rsquared_cs is None:
        rsquared_cs = r2cs_from_cstat(c_statistic, prevalence)

    max_r2_cs = max_r2_cs_binary(prevalence)
    if rsquared_cs >= max_r2_cs:
        raise ValueError(
            f"Anticipated Cox-Snell R² ({rsquared_cs:.4f}) must be smaller than "
            f"the maximum achievable for prevalence {prevalence:.3f} ({max_r2_cs:.4f})."
        )

    # Criterion 1: small overall shrinkage
    n1 = parameters / ((shrinkage - 1.0) * math.log(1.0 - rsquared_cs / shrinkage))
    n1 = math.ceil(n1)

    # Criterion 2: small absolute difference between apparent and adjusted R²_cs
    delta = 0.05
    s_vh = rsquared_cs / (rsquared_cs + delta * max_r2_cs)
    n2 = parameters / ((s_vh - 1.0) * math.log(1.0 - rsquared_cs / s_vh))
    n2 = math.ceil(n2)

    # Criterion 3: precise estimate of overall outcome proportion
    n3 = math.ceil((Z_975 / mape_prevalence) ** 2 * prevalence * (1.0 - prevalence))

    final_n = max(n1, n2, n3)
    final_events = math.ceil(final_n * prevalence)
    final_epp = final_events / parameters

    criteria = {
        "Criterion 1 — Global shrinkage": {
            "n": n1,
            "events": math.ceil(n1 * prevalence),
            "epp": (n1 * prevalence) / parameters,
            "explanation": f"Targets uniform shrinkage S = {shrinkage}.",
        },
        "Criterion 2 — Small optimism in R²_CS": {
            "n": n2,
            "events": math.ceil(n2 * prevalence),
            "epp": (n2 * prevalence) / parameters,
            "explanation": "Targets ≤0.05 absolute difference between apparent and adjusted Cox-Snell R².",
        },
        "Criterion 3 — Precision of outcome proportion": {
            "n": n3,
            "events": math.ceil(n3 * prevalence),
            "epp": (n3 * prevalence) / parameters,
            "explanation": f"Targets ±{mape_prevalence:.0%} margin of error on the overall outcome proportion.",
        },
    }

    return SampleSizeResult(
        outcome_type="binary",
        parameters=parameters,
        criteria=criteria,
        final_n=final_n,
        final_epp=final_epp,
        final_events=final_events,
        inputs={
            "prevalence": prevalence,
            "rsquared_cs": rsquared_cs,
            "c_statistic": c_statistic,
            "max_r2_cs": max_r2_cs,
            "shrinkage": shrinkage,
            "mape_prevalence": mape_prevalence,
        },
    )


# ---------------------------------------------------------------------------
# Survival (time-to-event) outcome
# ---------------------------------------------------------------------------

def pmsampsize_survival(
    parameters: int,
    rate: float,
    timepoint: float,
    meanfup: float,
    rsquared_cs: float,
    shrinkage: float = 0.9,
    mape_rate: float = 0.05,
) -> SampleSizeResult:
    """Sample size for a time-to-event prediction model (Riley 2019, Part II)."""
    if parameters < 1:
        raise ValueError("parameters must be ≥ 1.")
    if rate <= 0:
        raise ValueError("rate must be > 0 events per person-time.")
    if timepoint <= 0 or meanfup <= 0:
        raise ValueError("timepoint and meanfup must be > 0.")
    if not 0 < rsquared_cs < 1:
        raise ValueError("rsquared_cs must be in (0, 1).")
    if not 0 < shrinkage < 1:
        raise ValueError("shrinkage must be in (0, 1).")

    # For survival, max R²_cs = 1 - exp(-2 * D)/D approximation; use the
    # approximation from Riley 2019 Part II based on the overall event proportion
    # at mean follow-up.
    proportion_event = 1.0 - math.exp(-rate * meanfup)
    max_r2_cs = max_r2_cs_binary(proportion_event)

    if rsquared_cs >= max_r2_cs:
        raise ValueError(
            f"Anticipated Cox-Snell R² ({rsquared_cs:.4f}) must be < max ({max_r2_cs:.4f})."
        )

    # Criterion 1: shrinkage
    n1 = parameters / ((shrinkage - 1.0) * math.log(1.0 - rsquared_cs / shrinkage))
    n1 = math.ceil(n1)

    # Criterion 2: small optimism in R²_cs
    delta = 0.05
    s_vh = rsquared_cs / (rsquared_cs + delta * max_r2_cs)
    n2 = parameters / ((s_vh - 1.0) * math.log(1.0 - rsquared_cs / s_vh))
    n2 = math.ceil(n2)

    # Criterion 3: precise estimate of the overall event rate
    # n3 events such that the SE of log(rate) ≤ mape_rate / Z
    # Var(log rate) ≈ 1/E ⇒ E ≥ (Z/mape)²
    events_needed = math.ceil((Z_975 / mape_rate) ** 2)
    n3 = math.ceil(events_needed / proportion_event)

    final_n = max(n1, n2, n3)
    final_events = math.ceil(final_n * proportion_event)
    final_epp = final_events / parameters

    criteria = {
        "Criterion 1 — Global shrinkage": {
            "n": n1,
            "events": math.ceil(n1 * proportion_event),
            "epp": (n1 * proportion_event) / parameters,
            "explanation": f"Targets uniform shrinkage S = {shrinkage}.",
        },
        "Criterion 2 — Small optimism in R²_CS": {
            "n": n2,
            "events": math.ceil(n2 * proportion_event),
            "epp": (n2 * proportion_event) / parameters,
            "explanation": "Targets ≤0.05 absolute difference between apparent and adjusted Cox-Snell R².",
        },
        "Criterion 3 — Precision of event rate": {
            "n": n3,
            "events": math.ceil(n3 * proportion_event),
            "epp": (n3 * proportion_event) / parameters,
            "explanation": f"Targets ±{mape_rate:.0%} margin of error on log(rate).",
        },
    }

    return SampleSizeResult(
        outcome_type="survival",
        parameters=parameters,
        criteria=criteria,
        final_n=final_n,
        final_epp=final_epp,
        final_events=final_events,
        inputs={
            "rate": rate,
            "timepoint": timepoint,
            "meanfup": meanfup,
            "rsquared_cs": rsquared_cs,
            "max_r2_cs": max_r2_cs,
            "proportion_event": proportion_event,
            "shrinkage": shrinkage,
            "mape_rate": mape_rate,
        },
    )
