# Digital Twin — Stroke Rehabilitation

A Streamlit prototype that acts as a "digital twin" of a stroke-rehabilitation patient: you set the patient's admission characteristics with sliders, and the app predicts discharge outcomes (FIM, meaningful improvement, return home) from a simulated 200-patient cohort, then summarizes them as a composite 0–100 health score. It also includes a model-comparison benchmark (LOOCV) run on that same simulated cohort.

Everything runs on **simulated data** generated in code — no clinical data is needed or committed.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py
```

The first visit to "Model Comparison" runs the full LOOCV benchmark (a minute or two); results are cached afterwards.

## The two pages

The sidebar (in `app.py`) switches between:

1. **Digital Twin Prototype (mock)** — 20 sliders over a simulated 200-patient cohort, predictions via distance-weighted k-NN (k=20) → health gauge + outcome cards. The confidence boxplot is the spread of the 20 neighbors' composite scores.
2. **Model Comparison** — LOOCV benchmark of 3 algorithms × 3 feature sets on the same simulated cohort: metric heatmaps, confusion matrices, predicted-vs-actual scatter plots, best-model summary, and the rationale behind the health-score formula.

## Project structure

```
.
├── app.py                    # Streamlit entry point: page config, CSS, sidebar navigation
├── pyproject.toml            # Project metadata + dependencies (managed by uv)
├── uv.lock                   # Locked dependency versions (generated — do not edit)
├── .python-version           # Python version pin (3.12) for uv
├── README.md                 # This file
│
├── data/
│   └── mock_patients.py      # Synthetic 200-patient cohort generator + feature schema
│
├── engine/
│   ├── predictor.py          # Weighted k-NN predictor over the mock cohort
│   └── scoring.py            # Composite 0–100 health score
│
├── models/
│   ├── __init__.py           # Empty package marker (enables `models.*` imports)
│   ├── config.py             # Feature sets, targets, algorithms, display labels/colors
│   ├── train.py              # Model factory + LOOCV training pipeline
│   ├── evaluate.py           # Regression/classification metric helpers
│   └── results.py            # Streamlit-cached benchmark data + LOOCV results
│
└── ui/
    ├── sliders.py            # Vertical slider grid (inline Streamlit component)
    ├── health_gauge.py       # Health-score gauge: gradient bar + confidence boxplot
    ├── target_cards.py       # Predicted-outcome cards
    └── page_model_comparison.py  # "Model Comparison" page
```

## File-by-file guide

### Root

- **`app.py`** — the only entry point (`streamlit run app.py`). Sets the page config, injects the global medical-style CSS, and renders the sidebar radio that dispatches to the two pages. Each page's imports are lazy (inside its `if` branch). The mock page wires: `generate_cohort` → `render_sliders` → `predict` → `compute_health_score` → `render_health_gauge` + `render_target_cards`.
- **`pyproject.toml`** — project name, Python requirement (>=3.12), and dependencies.
- **`uv.lock`** / **`.python-version`** — uv's lockfile and interpreter pin; keep both committed for reproducible installs, never edit by hand.

### `data/` — simulated cohort

- **`mock_patients.py`** — `generate_cohort(n=200, seed=42)` builds a simulated cohort from a latent severity variable (20 clinical admission features + discharge outcomes: FIM total/motor/cognitive, FIM gain, meaningful improvement, return home). Also defines `FEATURE_NAMES`, `CATEGORICAL_MAP`, `FEATURE_DEFS` (the slider schema), and `to_model_frame(cohort)`, which encodes categoricals and renames columns to the `T0__*` / `OUTCOME__*` schema used by the benchmark (`MODEL_COLUMNS`).

### `engine/` — prediction and scoring logic

- **`predictor.py`** — `predict` encodes the slider input and cohort, computes inverse-distance-weighted averages over the k=20 nearest neighbors, and returns predictions plus the neighbor rows/weights (used for the confidence boxplot).
- **`scoring.py`** — `compute_health_score`: FIM discharge 40% / gain 25% / meaningful-improvement 20% / return-home 15%, with a true per-neighbor boxplot. Returns the `score/median/q1/q3/whisker_*/confidence` dict consumed by `ui/health_gauge.py`.

### `models/` — ML benchmark

- **`config.py`** — pure constants: the three feature subsets (`FEATURES_SMALL`/`MEDIUM`/`FULL` over `T0__*` columns, grouped in `FEATURE_SETS`), the 3 regression targets and the classification target, `MEANINGFUL_THRESHOLD = 22` (FIM-gain cutoff), `ALGORITHMS = ['knn', 'ridge', 'random_forest']`, and the display labels/colors used by the UI.
- **`train.py`** — `get_model(algo, task)` factory (k-NN k=5 distance-weighted; Ridge/LogisticRegression; RandomForest 100×depth-3, seed 42), `run_loocv` (leave-one-out CV with a per-fold `StandardScaler`), and `run_full_pipeline` (every feature-set × algorithm × target combination, plus the derived meaningful-improvement classification), returning a results dict keyed by `(feature_set, algo, target)`.
- **`evaluate.py`** — metric helpers on top of sklearn: MAE/RMSE/R² for regression; Accuracy/F1/Sensitivity/Specificity + confusion matrix for classification.
- **`results.py`** — Streamlit caching layer: `load_data()` (the mock cohort via `to_model_frame`, same seed as the mock page) and `get_all_results()` (full LOOCV pipeline).
- **`__init__.py`** — empty, just makes `models/` an importable package.

### `ui/` — Streamlit rendering

- **`sliders.py`** — `render_sliders(feature_defs, key_prefix)`: lays out vertical sliders 7 per row and returns `{feature name: value}` (option label for categorical features). The slider itself is an inline Streamlit Components v2 widget (HTML/CSS/JS defined in this file).
- **`health_gauge.py`** — `render_health_gauge(health)`: Plotly red-yellow-green gradient bar with the boxplot confidence overlay and a diamond marker at the composite score, plus the "score/100 ± confidence" panel.
- **`target_cards.py`** — `render_target_cards`: the 4 outcome cards under the gauge, with traffic-light coloring of probabilities.
- **`page_model_comparison.py`** — `render_model_comparison()`: MAE/R² heatmap tables, 3×3 confusion-matrix grids (Return Home + derived meaningful-improvement tabs), predicted-vs-actual scatter plots with patient hover, best-model summary, and the health-score methodology notes.

## Removed files

All recoverable from git history:

- October 2026: the real-patient "Digital Twin" page (`data/real_features.py`, `engine/real_predictor.py`, real-model scoring and cards) and the "Sample Size (pmsampsize)" page (`engine/pmsampsize.py`, `ui/page_pmsampsize.py`). The mock T0/T1/OUTCOME horizon selector was also removed.
- July 2026: `main.py` (uv hello-world stub), `ui/vertical_slider_component/` (legacy slider), `DigitalTWIN_Claude.md` and `PROMPT_MODEL_IMPLEMENTATION.md` (implemented specification prompts).
