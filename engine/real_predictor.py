"""Predictor using the best models from the LOOCV comparison, trained on all 20 real patients."""

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.config import (
    FEATURE_SETS,
    ALGORITHMS,
    REGRESSION_TARGETS,
    CLASSIFICATION_TARGET,
    FIM_ENTRY_COL,
    MEANINGFUL_THRESHOLD,
)
from models.train import get_model


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _find_best_models(results: dict) -> dict:
    """From LOOCV results, pick best (feature_set, algo) per target.

    Regression: lowest MAE.  Classification: highest F1.
    Returns dict  target -> (feature_set_name, algo).
    """
    best = {}

    for target in REGRESSION_TARGETS:
        best_key, best_mae = None, float("inf")
        for fs_name in FEATURE_SETS:
            for algo in ALGORITHMS:
                r = results.get((fs_name, algo, target))
                if r and r["MAE"] < best_mae:
                    best_mae = r["MAE"]
                    best_key = (fs_name, algo)
        if best_key:
            best[target] = best_key

    for target in [CLASSIFICATION_TARGET]:
        best_key, best_f1 = None, -1.0
        for fs_name in FEATURE_SETS:
            for algo in ALGORITHMS:
                r = results.get((fs_name, algo, target))
                if r and r["F1"] > best_f1:
                    best_f1 = r["F1"]
                    best_key = (fs_name, algo)
        if best_key:
            best[target] = best_key

    return best


def train_best_models(df: pd.DataFrame, results: dict):
    """Train the best model per target on ALL 20 patients.

    Returns dict with trained models, scalers, and feature columns per target.
    """
    best_choices = _find_best_models(results)
    all_t0 = [col for col in df.columns if col.startswith("T0__")]
    trained = {}

    for target, (fs_name, algo) in best_choices.items():
        fs_cols = FEATURE_SETS[fs_name]
        cols = all_t0 if fs_cols is None else fs_cols

        X = df[cols].values
        task = "classification" if target == CLASSIFICATION_TARGET else "regression"
        y = df[target].values
        if task == "classification":
            y = y.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = get_model(algo, task)
        model.fit(X_scaled, y)

        loocv = results.get((fs_name, algo, target), {})
        rmse = loocv.get("RMSE") if task == "regression" else None

        trained[target] = {
            "model": model,
            "scaler": scaler,
            "features": cols,
            "algo": algo,
            "feature_set": fs_name,
            "task": task,
            "rmse": rmse,
        }

    return trained


def predict_patient(patient_values: dict, trained_models: dict) -> dict:
    """Predict all targets for a single patient from slider values.

    patient_values: dict mapping T0__ column names to numeric values.
    Returns prediction dict with all outcomes.
    """
    predictions = {}

    for target, info in trained_models.items():
        features = info["features"]
        x = np.array([[patient_values.get(f, 0) for f in features]])
        x_scaled = info["scaler"].transform(x)
        model = info["model"]
        pred = model.predict(x_scaled)[0]
        predictions[target] = float(pred)

        if target == CLASSIFICATION_TARGET and hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", [0, 1]))
            proba = model.predict_proba(x_scaled)[0]
            pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
            predictions["return_home_prob"] = float(proba[pos_idx])

    # Derive FIM gain and meaningful improvement from FIM total prediction
    fim_total_target = REGRESSION_TARGETS[0]
    if fim_total_target in predictions:
        fim_entry = patient_values.get(FIM_ENTRY_COL, 0)
        fim_gain = predictions[fim_total_target] - fim_entry
        predictions["fim_gain"] = fim_gain
        predictions["meaningful_improvement"] = 1 if fim_gain >= MEANINGFUL_THRESHOLD else 0

        rmse = trained_models.get(fim_total_target, {}).get("rmse") or 0.0
        sigma = max(float(rmse), 1.0)
        predictions["meaningful_prob"] = 1.0 - _normal_cdf((MEANINGFUL_THRESHOLD - fim_gain) / sigma)

    if "return_home_prob" not in predictions and CLASSIFICATION_TARGET in predictions:
        predictions["return_home_prob"] = float(predictions[CLASSIFICATION_TARGET])

    return predictions
