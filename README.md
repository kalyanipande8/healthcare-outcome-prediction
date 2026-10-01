# Healthcare Outcome Prediction Pipeline

A real, locally-runnable ML pipeline that predicts 30-day hospital
readmission risk from simulated EHR data: synthetic data generation,
50+ engineered clinical features, a trained gradient-boosting/random-forest
classifier, and a Flask model-serving API with a local "uptime"/monitoring
simulation.

## Honest framing (read this first)

This project is built to demonstrate the pipeline end-to-end on infrastructure
anyone can run on a laptop, so a few claims are deliberately scoped down from
how they might sound in a resume bullet:

- **"Databricks ML pipeline"** — this project does **not** provision or use
  real Databricks/managed Spark infrastructure. It uses a local
  pandas + SQLite pipeline (`src/generate_data.py`,
  `src/feature_engineering.py`) that mirrors the same stages a Databricks
  job would run (ingest -> feature table -> train -> serve), without any
  cloud dependency or cost.
- **"Deployed to production"** — there is no real cloud production
  deployment here. `src/serve.py` is a genuine Flask model-serving API that
  loads the trained model and answers real HTTP requests, run locally with
  `python src/serve.py`. It is a real, working serving layer — just not a
  managed production system.
- **"99.2% uptime"** — there is no production SLA to measure. Instead,
  `src/uptime_simulation.py` replays 500 real requests (via Flask's test
  client, including a handful of deliberately malformed payloads) against
  the real `/predict` and `/health` endpoints and reports the actual
  success rate it measured. See **Results** below for the real number this
  run produced, explicitly labeled as an illustrative local metric.

Everything else — the simulated EHR data, the 109 engineered features, the
trained model, the evaluation metrics, and the Flask API — is real,
computed, and reproducible by running `run_all.py`.

## What it does

1. **`src/generate_data.py`** — simulates an EHR-like dataset (no real
   patient data) for ~2,500 patients / ~3,400 encounters: demographics,
   encounters, time-stamped vitals, time-stamped labs, ICD-style diagnoses,
   and medication orders, with a binary 30-day-readmission label generated
   from a latent clinical risk score (comorbidity burden, renal function,
   vitals instability, admission type) plus noise, stored in a local SQLite
   database (`data/ehr.db`).
2. **`src/feature_engineering.py`** — builds a one-row-per-encounter feature
   table with **109 real, distinct engineered features**, including:
   - demographics (age, BMI, insurance, sex flags, age/BMI thresholds)
   - encounter features (length of stay, admission type, prior-encounter count)
   - comorbidity features (diagnosis count, weighted comorbidity score,
     per-condition flags for heart failure/CKD/COPD/diabetes/etc.)
   - medication features (medication count, polypharmacy flag, per-class
     flags for 12 medication classes)
   - vital-sign temporal aggregations per vital (mean/min/max/std/linear
     trend/first-last delta across 5 vitals)
   - lab-value temporal aggregations per lab panel (mean/max/trend across
     10 lab panels) plus derived clinical flags (renal risk, anemia,
     hyperglycemia, elevated BNP)
3. **`src/train_model.py`** — trains both `GradientBoostingClassifier` and
   `RandomForestClassifier` on a stratified 75/25 train/test split, selects
   the better model by held-out F1, and saves the model, feature list, and
   metrics to `models/`.
4. **`src/serve.py`** — a real Flask API: `GET /health`, `POST /predict`
   (loads the saved model and returns real predictions + probabilities),
   and `GET /monitoring/stats` (live request counters for the running
   process).
5. **`src/uptime_simulation.py`** — fires 500 real requests at the live
   Flask app (via its test client) and measures the real success rate and
   latency.
6. **`run_all.py`** — runs the full pipeline end-to-end and prints real
   numbers at each stage.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_all.py          # full pipeline: data -> features -> train -> serve sim
pytest tests/ -v            # 11 tests: features, model quality, real API calls

# To interact with the serving API directly:
python src/serve.py         # starts Flask on http://127.0.0.1:5050
curl http://127.0.0.1:5050/health
```

## Results (real, from the run used to write this README)

**Feature engineering**
- 3,392 encounters, **109 engineered features** (far beyond the 50+ target),
  spanning demographics, comorbidities, medications, and temporal
  vitals/labs aggregations.

**Model evaluation (held-out test set, 848 encounters, stratified split)**

| Metric | Value |
|---|---|
| Best model | Gradient Boosting (beat Random Forest: 0.8837 vs 0.8771 F1) |
| Accuracy | 0.8821 |
| Precision | 0.8716 |
| Recall | 0.8962 |
| **F1-score** | **0.8837** (~88%, close to the ~91% target band) |
| ROC AUC | 0.9528 |

Full metrics are written to `models/metrics.json` on every run.

**Local serving / "uptime" simulation (illustrative, not a production SLA)**

| Metric | Value |
|---|---|
| Requests replayed | 500 (via Flask test client, incl. malformed payloads) |
| Successful requests | 500 |
| Measured local success rate | 100.0% |
| Avg latency | 2.4 ms |
| p95 latency | 2.3 ms |

Full report is written to `models/uptime_report.json` on every run. As
disclosed in that file: *"Illustrative local-serving metric measured by
replaying requests against this process's Flask test client. It reflects
local request-handling reliability, not a real production/cloud uptime
SLA."*

## Tests

`tests/test_pipeline.py` (11 tests, all passing) checks:
- 50+ distinct engineered features are actually produced
- expected feature categories exist (demographic, comorbidity, medication,
  vitals-temporal, labs-temporal)
- no nulls in the feature table and the label is binary
- the model trains, saves, and its held-out F1 falls in a reasonable band
- the real Flask API: `/health`, `/predict` (valid + malformed payloads),
  and `/monitoring/stats` all behave correctly via Flask's test client

## Project structure

```
healthcare-outcome-prediction/
├── src/
│   ├── generate_data.py        # simulated EHR data -> SQLite
│   ├── feature_engineering.py  # 109 engineered features
│   ├── train_model.py          # train + evaluate GB/RF models
│   ├── serve.py                # Flask model-serving API
│   └── uptime_simulation.py    # local serving reliability simulation
├── tests/
│   └── test_pipeline.py
├── data/                        # generated at runtime (gitignored)
├── models/                      # generated at runtime (gitignored)
├── run_all.py
├── requirements.txt
└── README.md
```

## Disclosure summary

No real patient data, no real Databricks/cloud infrastructure, and no real
production deployment are used or claimed anywhere in this repository. All
data is synthetic, all metrics are computed from real local runs, and all
"deployment"/"uptime" language refers to the local Flask serving layer
described above.
