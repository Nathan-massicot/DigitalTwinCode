"""Load data and cache pipeline results."""

import pandas as pd
import streamlit as st

from models.config import (
    FEATURE_SETS,
    ALGORITHMS,
    REGRESSION_TARGETS,
    CLASSIFICATION_TARGET,
    FIM_ENTRY_COL,
    FIM_GAIN_COL,
    MEANINGFUL_THRESHOLD,
)
from models.train import run_full_pipeline

DATA_PATH = "data/Feature_prototype_ML_ready.csv"


@st.cache_data
def load_real_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_all_results() -> dict:
    """Run the full pipeline and cache results."""
    df = load_real_data()
    return run_full_pipeline(
        df=df,
        feature_sets=FEATURE_SETS,
        algorithms=ALGORITHMS,
        regression_targets=REGRESSION_TARGETS,
        classification_target=CLASSIFICATION_TARGET,
        fim_entry_col=FIM_ENTRY_COL,
        fim_gain_col=FIM_GAIN_COL,
        meaningful_threshold=MEANINGFUL_THRESHOLD,
    )


@st.cache_resource
def get_trained_best_models() -> dict:
    """Train the best model per target on all 20 patients. Cached."""
    from engine.real_predictor import train_best_models
    df = load_real_data()
    results = get_all_results()
    return train_best_models(df, results)
