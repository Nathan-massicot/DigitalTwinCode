"""Load data and cache pipeline results."""

import pandas as pd
import streamlit as st

from data.mock_patients import generate_cohort, to_model_frame
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


@st.cache_data
def load_data() -> pd.DataFrame:
    """Same simulated cohort as the mock Digital Twin page, in benchmark schema."""
    return to_model_frame(generate_cohort(n=200, seed=42))


@st.cache_resource
def get_all_results() -> dict:
    """Run the full pipeline and cache results."""
    df = load_data()
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
