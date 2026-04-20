"""LOOCV training pipeline for all model combinations."""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_model(algo: str, task: str):
    """Return a fresh model instance."""
    if algo == "knn":
        if task == "regression":
            return KNeighborsRegressor(n_neighbors=5, weights="distance")
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    elif algo == "ridge":
        if task == "regression":
            return Ridge(alpha=1.0)
        return LogisticRegression(l1_ratio=0, solver="lbfgs", max_iter=1000)
    elif algo == "random_forest":
        if task == "regression":
            return RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        return RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    raise ValueError(f"Unknown algorithm: {algo}")


def run_loocv(X: np.ndarray, y: np.ndarray, algo: str, task: str):
    """Run LOOCV for a single model. Returns (y_true, y_pred) arrays."""
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = get_model(algo, task)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        y_true.append(y_test[0])
        y_pred.append(pred[0])

    return np.array(y_true), np.array(y_pred)


def run_full_pipeline(df: pd.DataFrame, feature_sets: dict, algorithms: list,
                      regression_targets: list, classification_target: str,
                      fim_entry_col: str, fim_gain_col: str,
                      meaningful_threshold: float):
    """
    Run all combinations: feature sets x algorithms x targets.
    Returns dict keyed by (feature_set_name, algo, target).
    """
    from models.evaluate import compute_regression_metrics, compute_classification_metrics

    results = {}
    all_t0 = [col for col in df.columns if col.startswith("T0__")]

    for fs_name, fs_cols in feature_sets.items():
        cols = all_t0 if fs_cols is None else fs_cols
        X = df[cols].values

        for algo in algorithms:
            # --- Regression targets ---
            for target in regression_targets:
                y = df[target].values
                y_true, y_pred = run_loocv(X, y, algo, "regression")
                metrics = compute_regression_metrics(y_true, y_pred)
                results[(fs_name, algo, target)] = {
                    "task": "regression",
                    "y_true": y_true,
                    "y_pred": y_pred,
                    **metrics,
                }

            # --- Classification: Return home ---
            y_cls = df[classification_target].values.astype(int)
            y_true_cls, y_pred_cls = run_loocv(X, y_cls, algo, "classification")
            cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)
            results[(fs_name, algo, classification_target)] = {
                "task": "classification",
                "y_true": y_true_cls,
                "y_pred": y_pred_cls,
                **cls_metrics,
            }

            # --- Derived: Meaningful improvement ---
            fim_total_key = (fs_name, algo, regression_targets[0])
            if fim_total_key in results:
                fim_pred = results[fim_total_key]["y_pred"]
                fim_entry = df[fim_entry_col].values
                gain_pred = fim_pred - fim_entry
                meaningful_pred = (gain_pred >= meaningful_threshold).astype(int)
                gain_true = df[fim_gain_col].values
                meaningful_true = (gain_true >= meaningful_threshold).astype(int)

                derived_metrics = compute_classification_metrics(meaningful_true, meaningful_pred)
                results[(fs_name, algo, "DERIVED__meaningful_improvement")] = {
                    "task": "classification_derived",
                    "y_true": meaningful_true,
                    "y_pred": meaningful_pred,
                    "gain_pred": gain_pred,
                    **derived_metrics,
                }

    return results
