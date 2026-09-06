"""Absolute project paths; never depend on the process working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BUNDLE_DIR = PROJECT_ROOT / "notebooks" / "fraud_dashboard_bundle"
ARTIFACT_DIR = BUNDLE_DIR / "artifacts"
RESULTS_DIR = BUNDLE_DIR / "results"
BUNDLE_DATA_DIR = BUNDLE_DIR / "data"

TRANSACTIONS_CSV = DATA_DIR / "transactions_clean.csv"
DASHBOARD_PARQUET = BUNDLE_DATA_DIR / "dashboard_data.parquet"
VALIDATION_DEMO_PARQUET = BUNDLE_DATA_DIR / "validation_demo.parquet"
