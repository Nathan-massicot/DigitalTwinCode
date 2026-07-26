# Digital Twin — Stroke Rehabilitation

A Streamlit prototype that acts as a "digital twin" of a stroke-rehabilitation patient: you set the patient's admission (T0) characteristics with sliders, and the app predicts discharge outcomes (FIM scores, meaningful improvement, return home) using models trained on a real 20-patient dataset, then summarizes them as a composite 0–100 health score. It also includes a model-comparison benchmark (LOOCV) and a sample-size calculator (a Python port of `pmsampsize`).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py
```

> **Note:** the real-patient dataset `data/Feature_prototype_ML_ready.csv` is deliberately **not committed** (clinical data, see `.gitignore`). The "Digital Twin" and "Model Comparison" pages need it; on a fresh clone you must obtain it separately and place it at that exact path. The app must be launched from the repo root, since the CSV path is relative.

## The four pages

The sidebar (in `app.py`) switches between four pages:

1. **Digital Twin** — the main page. 21 sliders for the real T0 features → best per-target models (selected by LOOCV benchmark, trained on the 20 real patients) predict discharge outcomes → health gauge + outcome cards.
2. **Digital Twin Prototype (mock)** — the original phase-1 prototype. 20 sliders over a simulated 200-patient cohort, predictions via distance-weighted k-NN (k=20), with a T0/T1/OUTCOME horizon selector that widens/narrows the confidence spread.
3. **Model Comparison** — LOOCV benchmark of 3 algorithms × 3 feature sets on the 20 real patients: metric heatmaps, confusion matrices, predicted-vs-actual scatter plots, best-model summary, and the rationale behind the health-score formula.
4. **Sample Size (pmsampsize)** — interactive Riley-criteria minimum-sample-size calculator for continuous, binary, and survival prediction models.

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
│   ├── Feature_prototype_ML_ready.csv   # Real 20-patient dataset (gitignored, clinical data)
│   ├── real_features.py      # Slider definitions for the 21 real T0 features
│   └── mock_patients.py      # Synthetic 200-patient cohort generator + mock feature schema
│
├── engine/
│   ├── real_predictor.py     # Selects + trains best models per target, predicts for slider input
│   ├── predictor.py          # Weighted k-NN predictor over the mock cohort
│   ├── scoring.py            # Composite 0–100 health score (real + mock variants)
│   └── pmsampsize.py         # Python port of the pmsampsize sample-size calculations
│
├── models/
│   ├── __init__.py           # Empty package marker (enables `models.*` imports)
│   ├── config.py             # Feature sets, targets, algorithms, display labels/colors
│   ├── train.py              # Model factory + LOOCV training pipeline
│   ├── evaluate.py           # Regression/classification metric helpers
│   └── results.py            # Streamlit-cached data loading, LOOCV results, trained models
│
└── ui/
    ├── sliders.py            # Vertical slider grid (inline Streamlit component)
    ├── health_gauge.py       # Health-score gauge: gradient bar + confidence boxplot
    ├── target_cards.py       # Predicted-outcome cards (mock + real variants)
    ├── page_model_comparison.py  # "Model Comparison" page
    └── page_pmsampsize.py    # "Sample Size (pmsampsize)" page
```

## File-by-file guide

### Root

- **`app.py`** — the only entry point (`streamlit run app.py`). Sets the page config, injects the global medical-style CSS, and renders the sidebar radio that dispatches to the four pages. Each page's imports are lazy (inside its `if` branch) so a page only loads its own dependencies. The main "Digital Twin" branch wires the whole flow: `REAL_FEATURE_DEFS` → `render_sliders` → `get_trained_best_models` → `predict_patient` → `compute_real_health_score` → `render_health_gauge` + `render_real_target_cards`.
- **`pyproject.toml`** — project name, Python requirement (>=3.12), and the six dependencies: streamlit, plotly, numpy, pandas, scikit-learn, matplotlib. No script entry points.
- **`uv.lock`** / **`.python-version`** — uv's lockfile and interpreter pin; keep both committed for reproducible installs, never edit by hand.

### `data/` — datasets and feature schemas

- **`Feature_prototype_ML_ready.csv`** *(gitignored)* — the real dataset: 20 patients × 29 columns (`patient_id`, 7 `OUTCOME__*` discharge targets, 21 `T0__*` admission features). Loaded by `models/results.py`; the 21 `T0__*` columns must stay in sync with `real_features.py` and `models/config.py`.
- **`real_features.py`** — `REAL_FEATURE_DEFS`: one dict per `T0__*` column (name, label, min/max/default, type, options) that drives the 21 sliders on the main page. This is the single place where the real feature space is described for the UI.
- **`mock_patients.py`** — everything for the mock page: `generate_cohort(n=200, seed=42)` builds a simulated cohort from a latent severity variable (20 clinical features + outcomes), plus `FEATURE_NAMES`, `CATEGORICAL_MAP`, and `FEATURE_DEFS` (the mock slider schema, used as the default by `ui/sliders.py`). The mock feature space (human-readable names) is entirely separate from the real `T0__*` space.

### `engine/` — prediction and scoring logic

- **`real_predictor.py`** — the real-model engine. `_find_best_models` picks, per target, the (feature set, algorithm) combo with lowest MAE (regression) or highest F1 (classification) from the LOOCV results; `train_best_models` refits those winners on all 20 patients with per-target scalers; `predict_patient` scales slider input, predicts each target, and derives FIM gain, meaningful improvement (gain ≥ threshold), and its probability via a normal CDF using the LOOCV RMSE as sigma.
- **`predictor.py`** — the mock engine: `predict` encodes the slider input and cohort, computes inverse-distance-weighted averages over the k=20 nearest neighbors, and returns predictions plus the neighbor rows/weights and a horizon-dependent `spread_factor` (T0=1.5, T1=1.0, OUTCOME=0.6) used for confidence spread.
- **`scoring.py`** — turns predictions into the gauge's data. `compute_real_health_score`: weighted composite of FIM Total 30% / Motor 20% / Cognitive 15% / Gain 20% / Return-home 15%, with spread estimated from the dispersion of the five components. `compute_health_score` (mock): FIM discharge 40% / gain 25% / meaningful-improvement 20% / return-home 15%, with a true per-neighbor boxplot rescaled by `spread_factor`. Both return the `score/median/q1/q3/whisker_*/confidence` dict consumed by `ui/health_gauge.py`.
- **`pmsampsize.py`** — self-contained port of the pmsampsize R/Stata package (Riley et al. 2019/2020): `pmsampsize_continuous`, `pmsampsize_binary`, `pmsampsize_survival` return a `SampleSizeResult` dataclass with per-criterion required N; helpers `r2cs_from_cstat` (Monte-Carlo C-statistic → Cox-Snell R²) and `max_r2_cs_binary`. No project-internal dependencies.

### `models/` — ML configuration, training, and caching

- **`config.py`** — pure constants, the single source of truth for the benchmark: the three feature subsets (`FEATURES_SMALL`/`MEDIUM`/`FULL` over `T0__*` columns, grouped in `FEATURE_SETS`), the 3 regression targets and the classification target, `MEANINGFUL_THRESHOLD = 22` (FIM-gain cutoff), `ALGORITHMS = ['knn', 'ridge', 'random_forest']`, and the display labels/colors used by the UI.
- **`train.py`** — `get_model(algo, task)` factory (k-NN k=5 distance-weighted; Ridge/LogisticRegression; RandomForest 100×depth-3, seed 42), `run_loocv` (leave-one-out CV with a per-fold `StandardScaler`), and `run_full_pipeline` (every feature-set × algorithm × target combination, plus the derived meaningful-improvement classification), returning a results dict keyed by `(feature_set, algo, target)`.
- **`evaluate.py`** — metric helpers on top of sklearn: MAE/RMSE/R² for regression; Accuracy/F1/Sensitivity/Specificity + confusion matrix for classification.
- **`results.py`** — the Streamlit caching layer between data/pipeline and pages: `load_real_data()` (`st.cache_data` read of the CSV at `DATA_PATH`), `get_all_results()` (`st.cache_resource` full LOOCV pipeline), `get_trained_best_models()` (`st.cache_resource` best models trained on all patients). This is the only module that reads the CSV.
- **`__init__.py`** — empty, just makes `models/` an importable package.

### `ui/` — Streamlit rendering

- **`sliders.py`** — `render_sliders(feature_defs, key_prefix)`: lays out vertical sliders 7 per row and returns `{feature name: value}`. The slider itself is an inline Streamlit Components v2 widget (HTML/CSS/JS defined in this file). Used by both twin pages: the mock page uses the default `FEATURE_DEFS`, the real page passes `REAL_FEATURE_DEFS`. Note: for categorical features it returns the option label in mock mode but the numeric index in real mode (keyed off `key_prefix`).
- **`health_gauge.py`** — `render_health_gauge(health)`: Plotly red-yellow-green gradient bar with the boxplot confidence overlay and a diamond marker at the composite score, plus the "score/100 ± confidence" panel.
- **`target_cards.py`** — the outcome cards under the gauge: `render_target_cards` (mock page, 4 cards) and `render_real_target_cards` (real page, 6 HTML cards: FIM Total/Motor/Cognitive, Gain, Meaningful Improvement, Return Home) with traffic-light coloring of probabilities.
- **`page_model_comparison.py`** — `render_model_comparison()`: renders the full LOOCV benchmark — MAE/R² heatmap tables, 3×3 confusion-matrix grids (Return Home + derived meaningful-improvement tabs), predicted-vs-actual scatter plots with patient hover, best-model summary, and the health-score methodology notes.
- **`page_pmsampsize.py`** — `render_pmsampsize()`: input forms for the continuous/binary/survival calculators, results as metric rows and per-criterion tables, plus methodological references. All math lives in `engine/pmsampsize.py`; this file is presentation only.

## How a prediction flows (main page)

```
data/real_features.py (REAL_FEATURE_DEFS)
        │ defines
        ▼
ui/sliders.py ──► patient_values ──► engine/real_predictor.predict_patient
                                          ▲ trained models
models/results.py ── loads CSV ──► models/train.run_full_pipeline (LOOCV)
                                          │ best per target
                                          ▼
                             engine/real_predictor.train_best_models
predictions ──► engine/scoring.compute_real_health_score ──► ui/health_gauge
           └──────────────────────────────────────────────► ui/target_cards
```

## Removed files (July 2026 cleanup)

These were deleted to simplify the repo (all recoverable from git history where tracked):

- `main.py` — leftover `uv init` hello-world stub, referenced by nothing.
- `ui/vertical_slider_component/` — legacy Components-v1 slider (`__init__.py` + `index.html`), superseded by the inline v2 component in `ui/sliders.py`; was never imported.
- `DigitalTWIN_Claude.md` — the original French phase-1 specification prompt; fully implemented, superseded by this README (still in git history).
- `PROMPT_MODEL_IMPLEMENTATION.md` — the phase-2 specification prompt for the ML pipeline; fully implemented, was untracked/gitignored.
