"""Rebuild active data/dashboard artifacts while preserving the frozen model."""
from __future__ import annotations

import argparse

from fraud_pipeline.cleaning import clean_transactions
from fraud_pipeline.data import load_raw_transactions
from fraud_pipeline.exports import (
    export_core_data,
    export_dashboard_artifacts,
    export_manifest,
    export_training_history,
)
from fraud_pipeline.features import engineer_features
from fraud_pipeline.splitting import chronological_split, summarize_splits
from fraud_pipeline.training import load_frozen_model_bundle


def build_compatibility_artifacts() -> dict:
    raw = load_raw_transactions()
    cleaned = clean_transactions(raw)
    splits = chronological_split(cleaned)
    engineered = engineer_features(cleaned)
    frozen = load_frozen_model_bundle()

    export_core_data(cleaned, splits)
    history_metadata = export_training_history(splits.train)
    validation_demo = export_dashboard_artifacts(engineered, frozen)
    manifest = export_manifest(cleaned, splits, history_metadata)

    print(summarize_splits(splits).to_string(index=False))
    print(f"Validation demo rows: {len(validation_demo):,}")
    print("Frozen Random Forest loaded; no fitting or model artifact writes occurred.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Explicitly request a new model (blocked in this structural-refactor release).",
    )
    args = parser.parse_args()
    if args.retrain:
        raise SystemExit(
            "Retraining is intentionally separated from compatibility builds. "
            "Run a reviewed model-methodology experiment before replacing frozen artifacts."
        )
    build_compatibility_artifacts()


if __name__ == "__main__":
    main()
