
# Fraud Dashboard Export Bundle

Final fraud model:
- Random Forest
- Selected features: 57
- Threshold: 0.793550

## artifacts/

- preprocessor.joblib
- best_rf_model.joblib
- confidence_model.joblib
- rf_feature_mask.npy
- processed_feature_names.json
- selected_feature_names.json
- confidence_feature_columns.json
- history_reference.json
- model_config.json

## data/

- dashboard_data.parquet
- validation_demo.parquet

## results/

Contains all available saved model-selection,
threshold, tuning, confidence, feature-importance,
and final-evaluation outputs.

IMPORTANT:
This bundle contains fitted artifacts.
The Streamlit application must not retrain models.
