# Fraud Detection & Risk Intelligence Dashboard

A Streamlit + Plotly presentation, analysis, and inference application built around the project's frozen fitted artifacts. The application does not retrain models, rerun feature selection, tune hyperparameters, optimize thresholds, or modify notebook outputs.

## Run locally

```bash
cd /path/to/deloitte
python -m pip install -r requirements.txt
streamlit run app.py
```

If your system Python has conflicting packages, use a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Pages

1. **Overview** — executive KPIs, final-test metrics, and model flow.
2. **Fraud Analysis** — page-wide filters and Plotly analysis across time, category, customer, and merchant.
3. **Model Development** — leakage-safe methodology, feature groups, selection results, and global importance.
4. **Model Performance** — frozen configuration, split comparison, confusion matrix, threshold analysis, and errors.
5. **Decision Evidence** — transaction-level Random Forest support and leakage-safe historical context.
6. **Transaction Demo** — single-row frozen inference, direct model evidence, training-history context, and separate ground truth.

## Architecture

```text
app.py                       Landing page
pages/                       Streamlit analysis pages
components/                  Reusable cards, charts, filters, and metrics
fraud_pipeline/              Reusable cleaning, splitting, features, model, and exports
utils/                       Canonical paths plus thin runtime adapters
data/fraud.csv               Canonical raw BankSim source
data/processed/              Generated cleaned parquet
data/splits/                 Generated chronological split parquets
notebooks/fraud_detection_analysis.ipynb
                              Active explanatory/orchestration notebook
notebooks/fraud_dashboard_bundle/
  artifacts/                 Frozen fitted sklearn objects and configuration
  data/                      Dashboard and validation-demo parquet exports
  results/                   Saved evaluation and analysis outputs
```

All paths are resolved from `utils/paths.py`, so launching the application does not depend on the current working directory. The dashboard requires the exported parquet for fast analysis and uses `@st.cache_data`; fitted sklearn artifacts use `@st.cache_resource`. See [`docs/PIPELINE.md`](docs/PIPELINE.md) for lineage, artifact ownership, and compatibility details.

Rebuild active data and dashboard artifacts without retraining:

```bash
python -m fraud_pipeline.build
```

## Dataset time semantics

`step` is the zero-based simulated day index. The BankSim dataset contains 180 simulated days (`0` through `179`), representing approximately six months. The dataset has no intraday timestamp, so hour-of-day cannot be derived from `step` or any other available field.

The frozen model was originally developed with legacy `day = step // 24` and `hour_of_day = step % 24` features based on an incorrect hourly interpretation. Those fields must not be presented as calendar day or hour-of-day. The current Random Forest feature-selection mask excludes both fields; fitted artifacts have not been replaced. The reproducible temporal comparison under `experiments/` evaluates the legacy baseline, a corrected direct-time representation, and a no-direct-time representation before any artifact replacement decision.

The UI uses an explicit high-contrast dark theme from `.streamlit/config.toml` and shared tokens in `components/styles.py`. Plotly figures use the same semantic colors and dark layout through one reusable helper.

For performance, Fraud Analysis filters the dataset once and caches compact chart-ready aggregates. Its amount histogram sends 120 pre-binned points to Plotly instead of nearly 595,000 raw records. The Transaction Demo browses a compact validation index, caps visible ID options at 100, and loads/scores one complete row at a time.

## Frozen inference flow

```text
89 raw engineered transaction features
→ saved preprocessor
→ saved RF feature mask (57 of 113 processed features)
→ saved Random Forest
→ fraud probability
→ frozen threshold
→ fraud / non-fraud decision
→ direct Random Forest evidence
   (signed margin, tree agreement and dispersion, reached-leaf support)
→ separate training-only historical context
```

Ground-truth fields and saved outputs are explicitly excluded from model inputs. The application does not estimate a probability that a prediction is correct and does not turn historical counts into normalized confidence scores.

Historical context is exported deterministically from the cleaned data by applying the same canonical chronological split used throughout the pipeline. `tools/export_training_history.py` is a thin wrapper around that shared split and export logic; it does not guess a boundary or read the dashboard artifact as a substitute. The five compact tables under `notebooks/fraud_dashboard_bundle/data/training_history/` contain raw entity counts, relationship counts, and amount summaries for 396,332 training transactions on simulated days 0–125. Fraud labels are excluded. Validation and test transactions are never incorporated into this reference.

Comparable historical outcomes are intentionally not shown. The frozen project does not define a validated, transparent similarity grouping, so adding a historical fraud rate would require an arbitrary new methodology. The output architecture retains an explicit unavailable state for future reviewed work.

The active analysis notebook is `notebooks/fraud_detection_analysis.ipynb`; reusable implementation lives in `fraud_pipeline/`. The retired original notebook is retained under `notebooks/legacy/` for provenance. This dashboard only loads exported data, results, and fitted artifacts. The older Logistic Regression confidence artifact, its legacy keys in `model_config.json`, and exported confidence-analysis results remain in the bundle solely as research history and are not used by application runtime.
