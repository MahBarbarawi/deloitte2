"""Single owner for processed, split, dashboard, and history exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from fraud_pipeline.config import (
    DASHBOARD_SCHEMA_VERSION,
    HISTORY_SCHEMA_VERSION,
    PIPELINE_VERSION,
    SPLIT_CONFIG,
)
from fraud_pipeline.features import model_dataset
from fraud_pipeline.history import build_training_history_reference
from fraud_pipeline.splitting import ChronologicalSplits, summarize_splits
from fraud_pipeline.training import FrozenModelBundle, score_with_frozen_model
from utils.paths import (
    BUNDLE_DATA_DIR,
    DASHBOARD_PARQUET,
    PIPELINE_MANIFEST,
    PROCESSED_TRANSACTIONS_PARQUET,
    PROJECT_ROOT,
    RAW_TRANSACTIONS_CSV,
    TEST_SPLIT_PARQUET,
    TRAINING_HISTORY_DIR,
    TRAIN_SPLIT_PARQUET,
    VALIDATION_DEMO_PARQUET,
    VALIDATION_SPLIT_PARQUET,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_core_data(cleaned: pd.DataFrame, splits: ChronologicalSplits) -> None:
    PROCESSED_TRANSACTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_SPLIT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(PROCESSED_TRANSACTIONS_PARQUET, index=False)
    splits.train.to_parquet(TRAIN_SPLIT_PARQUET, index=False)
    splits.validation.to_parquet(VALIDATION_SPLIT_PARQUET, index=False)
    splits.test.to_parquet(TEST_SPLIT_PARQUET, index=False)
    cleaned.to_parquet(DASHBOARD_PARQUET, index=False)


def export_training_history(train: pd.DataFrame) -> dict:
    tables = build_training_history_reference(train)
    TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_parquet(TRAINING_HISTORY_DIR / filename, index=False)
    metadata = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "purpose": "Label-free training history for dashboard Decision Evidence",
        "source": str(RAW_TRANSACTIONS_CSV.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(RAW_TRANSACTIONS_CSV),
        "split_definition": SPLIT_CONFIG.rule,
        "training_step_min": int(train["step"].min()),
        "training_step_max": int(train["step"].max()),
        "training_unique_steps": int(train["step"].nunique()),
        "training_row_count": len(train),
        "tables": {filename: len(frame) for filename, frame in tables.items()},
        "excluded_fields": ["fraud"],
    }
    (TRAINING_HISTORY_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def build_validation_demo(engineered: pd.DataFrame, bundle: FrozenModelBundle) -> pd.DataFrame:
    validation = engineered.loc[
        engineered["step"].between(SPLIT_CONFIG.validation_start, SPLIT_CONFIG.validation_end)
    ].copy()
    inputs = model_dataset(validation).drop(columns=["step", "fraud"])
    probabilities, predictions = score_with_frozen_model(inputs, bundle)
    output = inputs.reset_index(drop=True)
    output.insert(0, "source_index", validation.index.to_numpy())
    for column in ("step", "customer", "merchant", "zipcodeOri", "zipMerchant"):
        output[column] = validation[column].to_numpy()
    output["actual_fraud"] = validation["fraud"].to_numpy()
    output["prediction"] = predictions
    output["fraud_probability"] = probabilities
    return output


def export_dashboard_artifacts(engineered: pd.DataFrame, bundle: FrozenModelBundle) -> pd.DataFrame:
    validation_demo = build_validation_demo(engineered, bundle)
    validation_demo.to_parquet(VALIDATION_DEMO_PARQUET, index=False)
    return validation_demo


def export_manifest(cleaned: pd.DataFrame, splits: ChronologicalSplits, history_metadata: dict) -> dict:
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "dashboard_schema_version": DASHBOARD_SCHEMA_VERSION,
        "source": str(RAW_TRANSACTIONS_CSV.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(RAW_TRANSACTIONS_CSV),
        "dataset": {
            "rows": len(cleaned),
            "columns": cleaned.columns.tolist(),
            "simulation_day_min": int(cleaned["step"].min()),
            "simulation_day_max": int(cleaned["step"].max()),
            "unique_simulation_days": int(cleaned["step"].nunique()),
        },
        "split_definition": SPLIT_CONFIG.rule,
        "splits": summarize_splits(splits).to_dict(orient="records"),
        "history": history_metadata,
        "active_outputs": {
            "processed": str(PROCESSED_TRANSACTIONS_PARQUET.relative_to(PROJECT_ROOT)),
            "train_split": str(TRAIN_SPLIT_PARQUET.relative_to(PROJECT_ROOT)),
            "validation_split": str(VALIDATION_SPLIT_PARQUET.relative_to(PROJECT_ROOT)),
            "test_split": str(TEST_SPLIT_PARQUET.relative_to(PROJECT_ROOT)),
            "dashboard_data": str(DASHBOARD_PARQUET.relative_to(PROJECT_ROOT)),
            "validation_demo": str(VALIDATION_DEMO_PARQUET.relative_to(PROJECT_ROOT)),
            "training_history": str(TRAINING_HISTORY_DIR.relative_to(PROJECT_ROOT)),
        },
        "legacy_research_only": [
            "artifacts/confidence_model.joblib",
            "artifacts/confidence_feature_columns.json",
            "artifacts/history_reference.json",
            "results/confidence_summary.csv",
            "results/confidence_coefficients.csv",
            "results/confidence_results.csv",
            "results/confidence_by_prediction.csv",
        ],
    }
    PIPELINE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
