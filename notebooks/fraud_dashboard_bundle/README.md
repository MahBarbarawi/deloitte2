
# Fraud Dashboard Export Bundle

Final fraud model:
- Random Forest
- Selected features: 57
- Threshold: 0.793550

## artifacts/

- preprocessor.joblib
- best_rf_model.joblib
- confidence_model.joblib (retired confidence experiment; not used by dashboard runtime)
- rf_feature_mask.npy
- processed_feature_names.json
- selected_feature_names.json
- confidence_feature_columns.json (retired confidence experiment)
- history_reference.json (retired P95 normalization reference)
- model_config.json

## data/

- dashboard_data.parquet
- validation_demo.parquet
- training_history/ (schema-versioned, label-free reference from the canonical training split)

## pipeline_manifest.json

Records the raw-source hash, exact chronological split, active outputs, pipeline
versions, and research-only legacy artifacts.

## results/

Contains all available saved model-selection,
threshold, tuning, retired confidence experiment, feature-importance,
and final-evaluation outputs.

IMPORTANT:
This bundle contains fitted artifacts.
The Streamlit application must not retrain models.

Rebuild the active data bundle, history reference, and manifest from the project
root with `python -m fraud_pipeline.build`. This compatibility command loads but
does not modify the fitted model artifacts.
