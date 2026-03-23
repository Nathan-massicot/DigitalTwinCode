import numpy as np
import pandas as pd

from data.mock_patients import FEATURE_NAMES, CATEGORICAL_MAP


def _encode_features(df: pd.DataFrame) -> np.ndarray:
    """Encode features to numeric, min-max normalize."""
    encoded = []
    for feat in FEATURE_NAMES:
        col = df[feat]
        if feat in CATEGORICAL_MAP:
            options = CATEGORICAL_MAP[feat]
            col_numeric = col.map({v: i for i, v in enumerate(options)}).astype(float)
            col_min, col_max = 0, len(options) - 1
        else:
            col_numeric = col.astype(float)
            col_min, col_max = col_numeric.min(), col_numeric.max()
        if col_max - col_min > 0:
            col_numeric = (col_numeric - col_min) / (col_max - col_min)
        else:
            col_numeric = col_numeric * 0.0
        encoded.append(col_numeric.values)
    return np.column_stack(encoded)


def predict(patient_values: dict, cohort: pd.DataFrame, k: int = 20, horizon: str = "OUTCOME") -> dict:
    """Return predictions for the given patient against the cohort.

    horizon adjusts confidence spread:
      T0 -> wider spread, T1 -> medium, OUTCOME -> tighter.
    """
    patient_row = pd.DataFrame([patient_values])
    combined = pd.concat([cohort[FEATURE_NAMES], patient_row[FEATURE_NAMES]], ignore_index=True)

    # Compute min/max from cohort for normalization
    encoded_all = []
    for feat in FEATURE_NAMES:
        col = combined[feat]
        if feat in CATEGORICAL_MAP:
            options = CATEGORICAL_MAP[feat]
            col_numeric = col.map({v: i for i, v in enumerate(options)}).astype(float)
            col_min, col_max = 0, len(options) - 1
        else:
            col_numeric = col.astype(float)
            col_min = cohort[feat].astype(float).min()
            col_max = cohort[feat].astype(float).max()
        if col_max - col_min > 0:
            col_numeric = (col_numeric - col_min) / (col_max - col_min)
        else:
            col_numeric = col_numeric * 0.0
        encoded_all.append(col_numeric.values)

    matrix = np.column_stack(encoded_all)
    patient_vec = matrix[-1]
    cohort_matrix = matrix[:-1]

    distances = np.sqrt(np.sum((cohort_matrix - patient_vec) ** 2, axis=1))
    nearest_idx = np.argsort(distances)[:k]
    nearest_dist = distances[nearest_idx]
    nearest_dist = np.maximum(nearest_dist, 1e-8)

    weights = 1.0 / nearest_dist
    weights /= weights.sum()

    neighbors = cohort.iloc[nearest_idx]

    fim_discharge = float(np.average(neighbors["FIM Total at Discharge"], weights=weights))
    fim_gain = float(np.average(neighbors["FIM Gain"], weights=weights))
    meaningful_prob = float(np.average(neighbors["Meaningful Improvement"], weights=weights))
    return_home_prob = float(np.average(neighbors["_return_home_prob"], weights=weights))

    # Horizon affects confidence (spread multiplier)
    spread_factor = {"T0": 1.5, "T1": 1.0, "OUTCOME": 0.6}.get(horizon, 1.0)

    return {
        "fim_discharge": fim_discharge,
        "fim_gain": fim_gain,
        "meaningful_prob": meaningful_prob,
        "return_home_prob": return_home_prob,
        "neighbors": neighbors,
        "weights": weights,
        "spread_factor": spread_factor,
    }
