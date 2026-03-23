import numpy as np
import pandas as pd


def compute_health_score(predictions: dict) -> dict:
    """Compute composite health score 0-100 and confidence stats."""
    fim_d = predictions["fim_discharge"]
    fim_g = predictions["fim_gain"]
    m_prob = predictions["meaningful_prob"]
    rh_prob = predictions["return_home_prob"]
    neighbors = predictions["neighbors"]
    weights = predictions["weights"]
    spread_factor = predictions["spread_factor"]

    # Normalize components to 0-100
    fim_d_norm = np.clip((fim_d - 18) / (126 - 18) * 100, 0, 100)
    fim_g_norm = np.clip((fim_g - (-20)) / (80 - (-20)) * 100, 0, 100)
    m_norm = m_prob * 100
    rh_norm = rh_prob * 100

    score = 0.40 * fim_d_norm + 0.25 * fim_g_norm + 0.20 * m_norm + 0.15 * rh_norm

    # Compute per-neighbor scores for boxplot
    n_fim_d = np.clip((neighbors["FIM Total at Discharge"].values - 18) / (126 - 18) * 100, 0, 100)
    n_fim_g = np.clip((neighbors["FIM Gain"].values - (-20)) / (80 - (-20)) * 100, 0, 100)
    n_m = neighbors["Meaningful Improvement"].values * 100
    n_rh = neighbors["_return_home_prob"].values * 100

    neighbor_scores = 0.40 * n_fim_d + 0.25 * n_fim_g + 0.20 * n_m + 0.15 * n_rh

    median = float(np.median(neighbor_scores))
    q1 = float(np.percentile(neighbor_scores, 25))
    q3 = float(np.percentile(neighbor_scores, 75))
    s_min = float(np.min(neighbor_scores))
    s_max = float(np.max(neighbor_scores))

    # Apply spread factor: expand/contract around median
    def adjust(val):
        return float(np.clip(median + (val - median) * spread_factor, 0, 100))

    q1_adj = adjust(q1)
    q3_adj = adjust(q3)
    min_adj = adjust(s_min)
    max_adj = adjust(s_max)

    confidence = (q3_adj - q1_adj) / 2

    return {
        "score": float(np.clip(score, 0, 100)),
        "median": median,
        "q1": q1_adj,
        "q3": q3_adj,
        "whisker_low": min_adj,
        "whisker_high": max_adj,
        "confidence": confidence,
    }
