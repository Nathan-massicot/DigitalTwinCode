"""Configuration: feature subsets, targets, algorithms."""

# ---------------------------------------------------------------------------
# Feature subsets (adapted to actual CSV column names)
# "number of insurance claims" feature has been removed from the dataset.
# ---------------------------------------------------------------------------

FEATURES_SMALL = [
    "T0__fim_total_score_at_entry",
    "T0__fim_motor_subscale_at_entry",
    "T0__cirs_total_score",
    "T0__current_walking_distance_m",
    "T0__stroke_type_ischemic_hemorrhagic",
]

FEATURES_MEDIUM = FEATURES_SMALL + [
    "T0__fim_cognitive_subscale_at_entry",
    "T0__stroke_laterality_left_right_bilateral",
    "T0__thrombectomy_performed_yes_no",
    "T0__continence_status_at_entry",
    "T0__pre_stroke_walking_distance_m",
    "T0__stair_climbing_ability_at_entry",
    "T0__primary_diagnosis_icd",
]

FEATURES_FULL = None  # Resolved at runtime: all T0__ columns

FEATURE_SETS = {
    "Small (5)": FEATURES_SMALL,
    "Medium (12)": FEATURES_MEDIUM,
    "Full (21)": FEATURES_FULL,
}

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

REGRESSION_TARGETS = [
    "OUTCOME__fim_total_at_discharge",
    "OUTCOME__fim_motor_at_discharge_target",
    "OUTCOME__fim_cognitive_at_discharge_target",
]

CLASSIFICATION_TARGET = "OUTCOME__return_home_yes_no"

# Derived (not modelled):
#   FIM gain = predicted FIM total discharge − T0__fim_total_score_at_entry
#   Meaningful improvement = 1 if FIM gain >= MEANINGFUL_THRESHOLD else 0
# Reference columns for ground truth:
FIM_ENTRY_COL = "T0__fim_total_score_at_entry"
FIM_GAIN_COL = "OUTCOME__fim_gain_discharge__minus__entry"
MEANINGFUL_THRESHOLD = 22

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

ALGORITHMS = ["knn", "ridge", "random_forest"]

# Human-readable labels
TARGET_LABELS = {
    "OUTCOME__fim_total_at_discharge": "FIM Total",
    "OUTCOME__fim_motor_at_discharge_target": "FIM Motor",
    "OUTCOME__fim_cognitive_at_discharge_target": "FIM Cognitive",
    "OUTCOME__return_home_yes_no": "Return Home",
    "DERIVED__meaningful_improvement": "Meaningful Improvement",
}

ALGO_LABELS = {
    "knn": "k-NN",
    "ridge": "Ridge / Logistic",
    "random_forest": "Random Forest",
}

ALGO_COLORS = {
    "knn": "#3498db",
    "ridge": "#e67e22",
    "random_forest": "#2ecc71",
}
