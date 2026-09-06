# Fraud Detection & Risk Intelligence Dashboard

A Streamlit + Plotly presentation, analysis, and inference application built around the project's frozen fitted artifacts. The application does not retrain models, rerun feature selection, tune hyperparameters, optimize thresholds, or modify notebook outputs.

## Run locally

```bash
cd /Users/mahmoud/Documents/deloite
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
5. **Confidence** — correctness-model quality, calibration summaries, prediction-specific validation, and learned factors.
6. **Transaction Demo** — single-row frozen inference, behavioral context, fraud score, prediction confidence, evidence, and separate ground truth.

## Architecture

```text
app.py                       Landing page
pages/                       Streamlit analysis pages
components/                  Reusable cards, charts, filters, and metrics
utils/                       Absolute paths, cached loaders, inference, and formatting
data/transactions_clean.csv  Source cleaned EDA data
notebooks/fraud_dashboard_bundle/
  artifacts/                 Frozen fitted sklearn objects and configuration
  data/                      Dashboard and validation-demo parquet exports
  results/                   Saved evaluation and analysis outputs
```

All paths are resolved from `utils/paths.py`, so launching the application does not depend on the current working directory. The dashboard prefers the exported parquet for fast analysis and uses `@st.cache_data`; fitted sklearn artifacts use `@st.cache_resource`.

The UI uses an explicit high-contrast dark theme from `.streamlit/config.toml` and shared tokens in `components/styles.py`. Plotly figures use the same semantic colors and dark layout through one reusable helper.

For performance, Fraud Analysis filters the dataset once and caches compact chart-ready aggregates. Its amount histogram sends 120 pre-binned points to Plotly instead of nearly 595,000 raw records. The Transaction Demo browses a 14-column validation index, caps visible ID options at 100, and loads/scorers one complete row only when live inference is requested.

## Frozen inference flow

```text
89 raw engineered transaction features
→ saved preprocessor
→ saved RF feature mask (57 of 113 processed features)
→ saved Random Forest
→ fraud probability
→ frozen threshold
→ fraud / non-fraud decision
→ notebook-equivalent confidence features
→ saved Logistic Regression confidence model
→ probability that the final decision is correct
```

Ground-truth fields and saved outputs are explicitly excluded from model inputs. Training remains in `notebooks/Untitled.ipynb`; this dashboard only loads exported data, results, and fitted artifacts.
