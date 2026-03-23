import numpy as np
import pandas as pd

ICD_CODES = ["I63.0", "I63.1", "I63.3", "I63.4", "I63.5", "I61.0", "I61.1"]
LATERALITY_OPTIONS = ["Left", "Right", "Bilateral"]
STROKE_TYPE_OPTIONS = ["Ischemic", "Hemorrhagic"]


def generate_cohort(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Base severity factor per patient (0=mild, 1=severe)
    severity = rng.beta(2, 3, n)

    nihss = np.clip(np.round(severity * 42 + rng.normal(0, 3, n)), 0, 42).astype(int)

    fim_motor = np.clip(
        np.round(91 - severity * 70 + rng.normal(0, 8, n)), 13, 91
    ).astype(int)
    fim_cognitive = np.clip(
        np.round(35 - severity * 25 + rng.normal(0, 4, n)), 5, 35
    ).astype(int)
    fim_total = np.clip(fim_cognitive + fim_motor, 18, 126).astype(int)

    walking_distance = np.clip(
        np.round(500 * (1 - severity) ** 2 + rng.normal(0, 40, n)), 0, 500
    ).astype(int)
    pre_stroke_walking = np.clip(
        np.round(800 - severity * 300 + rng.normal(0, 100, n)), 0, 1000
    ).astype(int)

    laterality = rng.choice(LATERALITY_OPTIONS, n, p=[0.45, 0.45, 0.10])
    stroke_type = np.where(
        rng.random(n) < 0.15, "Hemorrhagic", "Ischemic"
    )
    thrombectomy = np.where(
        (stroke_type == "Ischemic") & (nihss >= 10) & (rng.random(n) < 0.4),
        "Yes", "No",
    )

    neglect = np.clip(np.round(severity * 3 + rng.normal(0, 0.5, n)), 0, 3).astype(int)
    aphasia = np.clip(np.round(severity * 2.5 + rng.normal(0, 0.6, n)), 0, 3).astype(int)
    arm_hemiparesis = np.clip(np.round(severity * 4 + rng.normal(0, 0.6, n)), 0, 4).astype(int)
    leg_hemiparesis = np.clip(np.round(severity * 3.5 + rng.normal(0, 0.6, n)), 0, 4).astype(int)
    walking_ability = np.clip(np.round(4 * (1 - severity) + rng.normal(0, 0.5, n)), 0, 4).astype(int)
    continence = np.clip(np.round(2 * (1 - severity) + rng.normal(0, 0.3, n)), 0, 2).astype(int)
    hygiene = np.clip(np.round(3 * (1 - severity) + rng.normal(0, 0.5, n)), 0, 3).astype(int)
    cirs = np.clip(np.round(severity * 40 + rng.normal(0, 6, n)), 0, 56).astype(int)
    stair_climbing = np.clip(np.round(3 * (1 - severity) + rng.normal(0, 0.5, n)), 0, 3).astype(int)
    orientation = np.clip(np.round(3 * (1 - severity * 0.6) + rng.normal(0, 0.4, n)), 0, 3).astype(int)

    icd = np.where(
        stroke_type == "Hemorrhagic",
        rng.choice(["I61.0", "I61.1"], n),
        rng.choice(["I63.0", "I63.1", "I63.3", "I63.4", "I63.5"], n),
    )

    # --- Outcomes ---
    recovery = (1 - severity) * 0.7 + rng.beta(2, 2, n) * 0.3

    fim_discharge = np.clip(
        np.round(fim_total + recovery * 50 + rng.normal(0, 5, n)), 18, 126
    ).astype(int)
    fim_gain = (fim_discharge - fim_total).astype(int)
    meaningful_improvement = (fim_gain >= 15).astype(int)

    return_home_prob = np.clip(
        0.3 + 0.5 * (fim_discharge / 126) + 0.1 * (walking_distance / 500)
        + rng.normal(0, 0.05, n),
        0, 1,
    )
    return_home = (rng.random(n) < return_home_prob).astype(int)

    df = pd.DataFrame({
        "FIM Cognitive Subscale": fim_cognitive,
        "FIM Motor Subscale": fim_motor,
        "FIM Total Score": fim_total,
        "NIHSS Total Score": nihss,
        "Walking Distance (m)": walking_distance,
        "Stroke Laterality": laterality,
        "Stroke Type": stroke_type,
        "Thrombectomy": thrombectomy,
        "Neglect Severity": neglect,
        "Aphasia Severity": aphasia,
        "Arm Hemiparesis": arm_hemiparesis,
        "Leg Hemiparesis": leg_hemiparesis,
        "Walking Ability": walking_ability,
        "Continence Status": continence,
        "Personal Hygiene": hygiene,
        "CIRS Total Score": cirs,
        "Primary Diagnosis (ICD)": icd,
        "Pre-stroke Walking (m)": pre_stroke_walking,
        "Stair Climbing": stair_climbing,
        "Orientation (time/place/person)": orientation,
        # outcomes
        "FIM Total at Discharge": fim_discharge,
        "FIM Gain": fim_gain,
        "Meaningful Improvement": meaningful_improvement,
        "Return Home": return_home,
        "_return_home_prob": return_home_prob,
        "_severity": severity,
    })
    return df


FEATURE_NAMES = [
    "FIM Cognitive Subscale",
    "FIM Motor Subscale",
    "FIM Total Score",
    "NIHSS Total Score",
    "Walking Distance (m)",
    "Stroke Laterality",
    "Stroke Type",
    "Thrombectomy",
    "Neglect Severity",
    "Aphasia Severity",
    "Arm Hemiparesis",
    "Leg Hemiparesis",
    "Walking Ability",
    "Continence Status",
    "Personal Hygiene",
    "CIRS Total Score",
    "Primary Diagnosis (ICD)",
    "Pre-stroke Walking (m)",
    "Stair Climbing",
    "Orientation (time/place/person)",
]

CATEGORICAL_MAP = {
    "Stroke Laterality": LATERALITY_OPTIONS,
    "Stroke Type": STROKE_TYPE_OPTIONS,
    "Thrombectomy": ["No", "Yes"],
    "Primary Diagnosis (ICD)": ICD_CODES,
}

FEATURE_DEFS = [
    {"name": "FIM Cognitive Subscale", "min": 5, "max": 35, "default": 22, "type": "numeric"},
    {"name": "FIM Motor Subscale", "min": 13, "max": 91, "default": 45, "type": "numeric"},
    {"name": "FIM Total Score", "min": 18, "max": 126, "default": 67, "type": "numeric"},
    {"name": "NIHSS Total Score", "min": 0, "max": 42, "default": 12, "type": "numeric"},
    {"name": "Walking Distance (m)", "min": 0, "max": 500, "default": 50, "type": "numeric"},
    {"name": "Stroke Laterality", "min": 0, "max": 2, "default": 0, "type": "categorical",
     "options": LATERALITY_OPTIONS},
    {"name": "Stroke Type", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": STROKE_TYPE_OPTIONS},
    {"name": "Thrombectomy", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["No", "Yes"]},
    {"name": "Neglect Severity", "min": 0, "max": 3, "default": 1, "type": "ordinal"},
    {"name": "Aphasia Severity", "min": 0, "max": 3, "default": 1, "type": "ordinal"},
    {"name": "Arm Hemiparesis", "min": 0, "max": 4, "default": 2, "type": "ordinal"},
    {"name": "Leg Hemiparesis", "min": 0, "max": 4, "default": 2, "type": "ordinal"},
    {"name": "Walking Ability", "min": 0, "max": 4, "default": 1, "type": "ordinal"},
    {"name": "Continence Status", "min": 0, "max": 2, "default": 1, "type": "ordinal"},
    {"name": "Personal Hygiene", "min": 0, "max": 3, "default": 1, "type": "ordinal"},
    {"name": "CIRS Total Score", "min": 0, "max": 56, "default": 8, "type": "numeric"},
    {"name": "Primary Diagnosis (ICD)", "min": 0, "max": 6, "default": 2, "type": "categorical",
     "options": ICD_CODES},
    {"name": "Pre-stroke Walking (m)", "min": 0, "max": 1000, "default": 500, "type": "numeric"},
    {"name": "Stair Climbing", "min": 0, "max": 3, "default": 1, "type": "ordinal"},
    {"name": "Orientation (time/place/person)", "min": 0, "max": 3, "default": 3, "type": "ordinal"},
]
