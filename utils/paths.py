"""Absolute project paths; never depend on the process working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "splits"
BUNDLE_DIR = PROJECT_ROOT / "notebooks" / "fraud_dashboard_bundle"
ARTIFACT_DIR = BUNDLE_DIR / "artifacts"
RESULTS_DIR = BUNDLE_DIR / "results"
BUNDLE_DATA_DIR = BUNDLE_DIR / "data"
BUNDLE_ARCHIVE = PROJECT_ROOT / "notebooks" / "fraud_dashboard_bundle.zip"
PIPELINE_MANIFEST = BUNDLE_DIR / "pipeline_manifest.json"
EXPERIMENT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

RAW_TRANSACTIONS_CSV = RAW_DATA_DIR / "fraud.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions_clean.csv"  # Legacy compatibility copy.
PROCESSED_TRANSACTIONS_PARQUET = PROCESSED_DATA_DIR / "transactions_clean.parquet"
TRAIN_SPLIT_PARQUET = SPLIT_DATA_DIR / "train.parquet"
VALIDATION_SPLIT_PARQUET = SPLIT_DATA_DIR / "validation.parquet"
TEST_SPLIT_PARQUET = SPLIT_DATA_DIR / "test.parquet"
DASHBOARD_PARQUET = BUNDLE_DATA_DIR / "dashboard_data.parquet"
DASHBOARD_CSV = BUNDLE_DATA_DIR / "dashboard_data.csv"
VALIDATION_DEMO_PARQUET = BUNDLE_DATA_DIR / "validation_demo.parquet"
TRAINING_HISTORY_DIR = BUNDLE_DATA_DIR / "training_history"
FROZEN_PREPROCESSOR = ARTIFACT_DIR / "preprocessor.joblib"
FROZEN_RANDOM_FOREST = ARTIFACT_DIR / "best_rf_model.joblib"
FROZEN_MODEL_CONFIG = ARTIFACT_DIR / "model_config.json"
SELECTED_FEATURE_NAMES = ARTIFACT_DIR / "selected_feature_names.json"
