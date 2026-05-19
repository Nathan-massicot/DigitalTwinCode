"""Feature definitions for the real 20-patient dataset (slider configs)."""

REAL_FEATURE_DEFS = [
    # --- Core FIM scores ---
    {"name": "T0__fim_total_score_at_entry", "label": "FIM Total", "min": 18, "max": 126, "default": 60, "type": "numeric"},
    {"name": "T0__fim_motor_subscale_at_entry", "label": "FIM Motor", "min": 13, "max": 91, "default": 40, "type": "numeric"},
    {"name": "T0__fim_cognitive_subscale_at_entry", "label": "FIM Cognitive", "min": 5, "max": 35, "default": 22, "type": "numeric"},
    # --- Stroke characteristics ---
    {"name": "T0__stroke_type_ischemic_hemorrhagic", "label": "Stroke Type", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["Ischemic", "Hemorrhagic"]},
    {"name": "T0__stroke_laterality_left_right_bilateral", "label": "Laterality", "min": 0, "max": 2, "default": 0, "type": "categorical",
     "options": ["Left", "Right", "Bilateral"]},
    {"name": "T0__thrombectomy_performed_yes_no", "label": "Thrombectomy", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["No", "Yes"]},
    {"name": "T0__primary_diagnosis_icd", "label": "Diagnosis ICD", "min": 0, "max": 1, "default": 0, "type": "numeric", "step": 0.1},
    # --- Mobility ---
    {"name": "T0__current_walking_distance_m", "label": "Walking Dist. (m)", "min": 0, "max": 900, "default": 10, "type": "numeric"},
    {"name": "T0__pre_stroke_walking_distance_m", "label": "Pre-stroke Walk (m)", "min": 0, "max": 1000, "default": 500, "type": "numeric"},
    {"name": "T0__stair_climbing_ability_at_entry", "label": "Stair Climbing", "min": 0, "max": 3, "default": 1, "type": "ordinal"},
    # --- Autonomy ---
    {"name": "T0__personal_hygiene_at_entry", "label": "Personal Hygiene", "min": 1, "max": 3, "default": 2, "type": "ordinal"},
    {"name": "T0__continence_status_at_entry", "label": "Continence", "min": 0, "max": 2, "default": 1, "type": "ordinal"},
    {"name": "T0__transfer_number_of_helpers_needed", "label": "Transfer Helpers", "min": 0, "max": 2, "default": 1, "type": "ordinal"},
    # --- Comorbidities ---
    {"name": "T0__cirs_total_score", "label": "CIRS Total", "min": 0, "max": 56, "default": 12, "type": "numeric"},
    # --- Pre-stroke history ---
    {"name": "T0__pre_stroke_specialist_count", "label": "Specialist Count", "min": 0, "max": 30, "default": 10, "type": "numeric"},
    {"name": "T0__pre_stroke_hospitalization_yes_no", "label": "Prior Hospitaliz.", "min": 0, "max": 1, "default": 1, "type": "categorical",
     "options": ["No", "Yes"]},
    {"name": "T0__pre_stroke_cardio_neuro_visits_yes_no", "label": "Cardio/Neuro Visits", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["No", "Yes"]},
    # --- Demographics ---
    {"name": "T0__sex_m_f", "label": "Sex", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["Male", "Female"]},
    # --- Medications ---
    {"name": "T0__analgesics_n02_present", "label": "Analgesics", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["No", "Yes"]},
    {"name": "T0__gastroprotection_ppi_a02_present", "label": "Gastroprotection", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["No", "Yes"]},
    {"name": "T0__laxatives_a06_present", "label": "Laxatives", "min": 0, "max": 1, "default": 0, "type": "categorical",
     "options": ["No", "Yes"]},
]
