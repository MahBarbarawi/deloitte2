# Fraud Pipeline

## Active lineage

The reproducible compatibility build is:

```text
data/fraud.csv
  -> fraud_pipeline.data.load_raw_transactions
  -> fraud_pipeline.cleaning.clean_transactions
  -> fraud_pipeline.splitting.chronological_split
  -> fraud_pipeline.features.engineer_features
  -> existing frozen preprocessor and Random Forest
  -> fraud_pipeline.exports
  -> dashboard, validation-demo, split, history, and manifest artifacts
```

Run it from the repository root with:

```bash
python -m fraud_pipeline.build
```

This is a compatibility build. It loads the fitted model but never fits or writes
model artifacts. `--retrain` is deliberately blocked until the separate temporal
methodology comparison has been reviewed.

## Dataset semantics

`step` is the zero-based simulated day index. The dataset covers 180 simulated
days, `0` through `179`, or approximately six months. BankSim provides no
intraday timestamp, so hour-of-day cannot be recovered. The legacy formulas
`day = step // 24` and `hour_of_day = step % 24` are invalid interpretations.

The canonical split is defined once in `fraud_pipeline/config.py`:

| Split | Simulated days | Rows | Fraud | Legitimate |
|---|---:|---:|---:|---:|
| Train | 0-125 | 396,332 | 5,040 | 391,292 |
| Validation | 126-152 | 98,194 | 1,080 | 97,114 |
| Test | 153-179 | 100,117 | 1,080 | 99,037 |

Historical model features for a transaction on simulated day T use only days
strictly earlier than T. Transactions within T cannot use one another because
the source has no reliable within-day order.

## Active artifacts

| Artifact | Producer | Consumer |
|---|---|---|
| `data/processed/transactions_clean.parquet` | compatibility build | inspection/reuse |
| `data/splits/{train,validation,test}.parquet` | compatibility build | reproducible split inspection |
| `notebooks/fraud_dashboard_bundle/data/dashboard_data.parquet` | compatibility build | Fraud Analysis and overview pages |
| `notebooks/fraud_dashboard_bundle/data/validation_demo.parquet` | compatibility build | Transaction Demo and Decision Evidence |
| `notebooks/fraud_dashboard_bundle/data/training_history/*.parquet` | compatibility build or thin export wrapper | Decision Evidence historical context |
| `notebooks/fraud_dashboard_bundle/data/training_history/metadata.json` | compatibility build or thin export wrapper | schema/split validation at runtime |
| `notebooks/fraud_dashboard_bundle/pipeline_manifest.json` | compatibility build | lineage and output inventory |
| `notebooks/fraud_dashboard_bundle/artifacts/preprocessor.joblib` | legacy model training | frozen inference |
| `notebooks/fraud_dashboard_bundle/artifacts/best_rf_model.joblib` | legacy model training | frozen inference |
| `notebooks/fraud_dashboard_bundle/artifacts/selected_feature_names.json` | legacy model training | frozen compatibility selection |

All active path definitions live in `utils/paths.py`. Dashboard runtime only
loads generated artifacts; it does not clean, split, engineer a full dataset, or
build training history.

## Frozen-model compatibility

The saved preprocessor schema still contains the invalid legacy `day` and
`hour_of_day` inputs. Neither was selected for the 57-feature Random Forest.
The compatibility adapter supplies zero placeholders to satisfy that old
preprocessor schema, then selects the same 57 processed features in the same
order. This preserves frozen probabilities and decisions while keeping invalid
temporal fields out of active exported data and UI semantics.

The corrupt legacy `rf_feature_mask.npy` is retained as historical evidence;
runtime derives the mask from the authoritative selected-feature names and
validates the resulting 57-column contract. No fitted artifact is replaced.

## Training-history schema

Schema version 2 is label-free and is built only from the canonical training
split. It contains:

- customer, merchant, and category transaction counts plus amount mean/min/max;
- customer-merchant and customer-category transaction counts;
- metadata with source hash, split definition, row count, day range, schema
  version, and pipeline version.

It intentionally excludes `fraud`. Historical Context is factual training-set
experience, not another model and not a probability of correctness.

## Notebook and legacy research

`notebooks/fraud_detection_analysis.ipynb` is the active explanatory notebook.
It imports pipeline functions rather than owning duplicate implementation.
The former notebook is preserved at
`notebooks/legacy/retired_fraud_detection_experiment.ipynb` and is explicitly
marked retired because it contains the invalid time encoding and old Logistic
Regression confidence experiment.

The following byte-identical cleaned CSV copies remain for compatibility or
manual inspection, but are not active generation sources:

- `data/transactions_clean.csv`
- `notebooks/transactions_clean.csv`
- `notebooks/fraud_dashboard_bundle/data/dashboard_data.csv`

The CSV in the dashboard bundle is inspection-only; Streamlit continues to use
Parquet. Legacy confidence model/results and the historical ZIP are retained as
research/provenance artifacts and are listed under `legacy_research_only` in the
pipeline manifest.
