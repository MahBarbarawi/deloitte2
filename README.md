# Fraud Detection & Risk Intelligence

This repository is intended to contain a transaction-level fraud-detection system built using approximately 594,643 transactions.

The system includes exploratory fraud analysis, leakage-safe historical behavioral feature engineering, preprocessing, feature selection, a tuned Random Forest classifier, a custom fraud decision threshold, a learned prediction-confidence layer, and a Streamlit + Plotly presentation dashboard.

Training and experimentation live in `notebooks/Untitled.ipynb`. The Streamlit application does **not** retrain the model: it loads frozen fitted artifacts exported from the notebook under `notebooks/fraud_dashboard_bundle/`.

> **Current checkout status:** the application, notebook, utility modules, datasets, and exported model bundle in the current Git commit are all 0-byte files. The directory structure is present, but the dashboard and saved model cannot run until the real file contents are restored. In particular, there is currently no callable inference API in `utils/inference.py` or artifact-loading API in `utils/artifact_loader.py` to document safely.

## 1. Quick start

After restoring the application and model-bundle files:

```bash
cd /Users/mahmoud/Documents/deloitte

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -m streamlit run app.py
```

Streamlit normally prints a local address such as `http://localhost:8501`.

## 2. How the fraud-detection system works

The intended scoring flow is:

```text
transaction + prior behavioral history
  -> fitted preprocessing
  -> selected features
  -> tuned Random Forest
  -> fraud probability
  -> custom decision threshold
  -> fraud / non-fraud prediction
  -> learned prediction confidence
```

Historical behavioral features must use only information available before the transaction being scored. Using future transactions would leak information into the model and make its reported performance unreliable.

The custom threshold exported by the notebook must be used for the final class decision. Do not silently replace it with scikit-learn's usual `0.5` cutoff.

## 3. Repository structure

```text
app.py
pages/
  1_Overview.py
  2_Fraud_Analysis.py
  3_Model_Development.py
  4_Model_Performance.py
  5_Confidence.py
  6_Transaction_Demo.py
components/
  cards.py
  charts.py
  filters.py
  metrics.py
  styles.py
utils/
  artifact_loader.py
  confidence.py
  data_loader.py
  formatting.py
  inference.py
  paths.py
data/
notebooks/
  Untitled.ipynb
  fraud_dashboard_bundle/
    artifacts/
      best_rf_model.joblib
      confidence_model.joblib
      preprocessor.joblib
      rf_feature_mask.npy
      model_config.json
      history_reference.json
      confidence_feature_columns.json
      processed_feature_names.json
      selected_feature_names.json
    data/
      dashboard_data.parquet
      validation_demo.parquet
    results/
      final_metrics.json
      threshold_results.csv
      tuning_results.csv
      feature_selection_results.csv
      rf_feature_importance.csv
      confidence_results.csv
      confidence_summary.csv
      confidence_coefficients.csv
      confidence_by_prediction.csv
      validation_vs_test.csv
requirements.txt
```

The notebook owns training and artifact export. The dashboard pages present the saved data, results, and predictions. The `components/` directory is for shared presentation code, while `utils/` is intended to hold data loading, artifact loading, inference, confidence, formatting, and path helpers.

## 4. Load the saved model

A verified loading example cannot be provided from this checkout because `utils/artifact_loader.py` and every file in `notebooks/fraud_dashboard_bundle/artifacts/` are empty. Guessing a loader function or configuration schema here would create an API that does not exist.

Once the real files are restored, use the public loader defined in `utils/artifact_loader.py`. Keep these exported parts together as one fitted scoring pipeline:

- the preprocessor
- the Random Forest feature mask
- the tuned Random Forest model
- the saved model configuration and decision threshold
- the confidence model and its expected feature columns
- the historical reference data needed by inference

Do not retrain, refit, or reconstruct these objects in dashboard code.

## 5. Make one prediction

A correct runnable prediction example also depends on the missing implementation in `utils/inference.py`. After restoration, the example should call the real public inference function from that module with one complete transaction row, preferably from `notebooks/fraud_dashboard_bundle/data/validation_demo.parquet`.

The scoring code must preserve the trained pipeline's:

- required raw input columns and order
- feature meanings, units, and data types
- leakage-safe historical feature calculations
- fitted preprocessing
- saved feature mask
- custom decision threshold
- confidence-feature construction

Do not pass target labels, future activity, saved predictions, or other output columns into the fraud model.

## 6. Fraud probability vs prediction confidence

These values answer different questions:

| Value | What it means |
|---|---|
| **Fraud probability** | The Random Forest score for the transaction belonging to the fraud class. The saved custom threshold converts this score into `fraud` or `non-fraud`. |
| **Prediction confidence** | The learned estimate of how trustworthy the final class decision is, based on the separate confidence layer. |

A transaction can have a high-looking fraud probability and still be classified as non-fraud when its score is below the saved custom threshold. Its prediction confidence is a separate output and must be calculated by the exported confidence logic—not inferred as `max(p, 1 - p)` or treated as another name for fraud probability.

In short: **fraud probability describes fraud risk; prediction confidence describes trust in the final prediction.**
