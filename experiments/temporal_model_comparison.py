"""Compare legacy and corrected BankSim temporal representations.

This experiment deliberately does not update or replace fitted model artifacts.
It reuses the canonical fraud_pipeline historical feature engineering, then reruns
the downstream train-only preprocessing, feature
selection, model selection, threshold selection, and final test evaluation for
three temporal variants.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from fraud_pipeline.cleaning import clean_transactions
from fraud_pipeline.config import SPLIT_CONFIG
from fraud_pipeline.data import load_raw_transactions
from fraud_pipeline.evaluation import best_f1_threshold, evaluate_probabilities
from fraud_pipeline.features import engineer_features, model_dataset
from fraud_pipeline.splitting import chronological_split
from fraud_pipeline.training import (
    build_preprocessor,
    select_random_forest_features,
    train_random_forest,
)
from utils.paths import ARTIFACT_DIR, EXPERIMENT_RESULTS_DIR, RAW_TRANSACTIONS_CSV

SOURCE_PATH = RAW_TRANSACTIONS_CSV
DEFAULT_OUTPUT_DIR = EXPERIMENT_RESULTS_DIR / "temporal_model_comparison"

RANDOM_STATE = 42
TUNING_CONFIG = {
    "n_iterations": 25,
    "comparison_threshold": 0.50,
    "max_f1_gap": 0.08,
    "gap_penalty": 0.50,
    "random_state": RANDOM_STATE,
}

PARAMETER_SPACE = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [6, 8, 10, 12, 15, 20],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.5, 0.75],
    "max_samples": [0.60, 0.75, 0.90, None],
}


def directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def build_engineered_data() -> pd.DataFrame:
    cleaned = clean_transactions(load_raw_transactions())
    engineered = engineer_features(cleaned)
    expected_rows = len(cleaned)
    if len(engineered) != expected_rows or engineered["step"].min() != 0 or engineered["step"].max() != 179:
        raise RuntimeError("Canonical feature engineering produced an unexpected population or step range")
    return engineered


def temporal_variant(engineered: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "legacy_baseline":
        # Research reproduction only. These encodings are semantically invalid
        # and are never created by the active pipeline or dashboard.
        engineered = engineered.copy()
        legacy_position = engineered.columns.get_loc("fraud") + 1
        engineered.insert(legacy_position, "day", engineered["step"] // 24)
        engineered.insert(legacy_position + 1, "hour_of_day", engineered["step"] % 24)
    data = model_dataset(engineered)
    if variant == "corrected_simulation_day":
        data["simulation_day"] = data["step"]
    elif variant not in {"legacy_baseline", "no_direct_time"}:
        raise ValueError(f"Unknown variant: {variant}")
    return data


def split_features(data: pd.DataFrame):
    splits = chronological_split(data)

    def xy(frame: pd.DataFrame):
        return frame.drop(columns=["fraud", "step"]), frame["fraud"].astype(int)

    return (*xy(splits.train), *xy(splits.validation), *xy(splits.test))


def run_variant(engineered: pd.DataFrame, variant: str):
    started = time.monotonic()
    print(f"[{variant}] preparing data", flush=True)
    data = temporal_variant(engineered, variant)
    X_train, y_train, X_validation, y_validation, X_test, y_test = split_features(data)
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)
    expected_processed_counts = {
        "legacy_baseline": 113,
        "corrected_simulation_day": 112,
        "no_direct_time": 111,
    }
    if X_train_processed.shape[1] != expected_processed_counts[variant]:
        raise RuntimeError(
            f"{variant} produced {X_train_processed.shape[1]} processed features; "
            f"expected {expected_processed_counts[variant]}"
        )

    print(f"[{variant}] fitting train-only RF feature selector", flush=True)
    selector = select_random_forest_features(X_train_processed, y_train)
    selected_mask = selector.get_support()
    processed_names = preprocessor.get_feature_names_out()
    selected_names = processed_names[selected_mask]
    X_train_selected = X_train_processed[:, selected_mask]
    X_validation_selected = X_validation_processed[:, selected_mask]
    X_test_selected = X_test_processed[:, selected_mask]

    selector_importances = pd.DataFrame(
        {
            "variant": variant,
            "feature": processed_names,
            "selector_importance": selector.estimator_.feature_importances_,
            "selected": selected_mask,
        }
    ).sort_values("selector_importance", ascending=False)

    parameter_combinations = list(
        ParameterSampler(
            PARAMETER_SPACE,
            n_iter=TUNING_CONFIG["n_iterations"],
            random_state=TUNING_CONFIG["random_state"],
        )
    )
    tuning_rows = []
    for iteration, params in enumerate(parameter_combinations, start=1):
        print(f"[{variant}] tuning {iteration:02d}/{len(parameter_combinations)}", flush=True)
        model = train_random_forest(X_train_selected, y_train, bootstrap=True, **params)
        train_probability = model.predict_proba(X_train_selected)[:, 1]
        validation_probability = model.predict_proba(X_validation_selected)[:, 1]
        train_metrics = evaluate_probabilities(
            y_train, train_probability, TUNING_CONFIG["comparison_threshold"]
        )
        validation_metrics = evaluate_probabilities(
            y_validation, validation_probability, TUNING_CONFIG["comparison_threshold"]
        )
        f1_gap = train_metrics["f1"] - validation_metrics["f1"]
        pr_auc_gap = train_metrics["pr_auc"] - validation_metrics["pr_auc"]
        tuning_rows.append(
            {
                "variant": variant,
                "iteration": iteration,
                **params,
                **{f"train_{key}": value for key, value in train_metrics.items() if key not in {"tn", "fp", "fn", "tp"}},
                **{f"validation_{key}": value for key, value in validation_metrics.items() if key not in {"tn", "fp", "fn", "tp"}},
                "f1_gap": f1_gap,
                "pr_auc_gap": pr_auc_gap,
                "selection_score": validation_metrics["pr_auc"]
                - TUNING_CONFIG["gap_penalty"] * max(f1_gap, 0),
            }
        )
    tuning = pd.DataFrame(tuning_rows)
    acceptable = tuning[tuning["f1_gap"] <= TUNING_CONFIG["max_f1_gap"]]
    if acceptable.empty:
        chosen = tuning.sort_values("selection_score", ascending=False).iloc[0]
    else:
        chosen = acceptable.sort_values(
            ["validation_pr_auc", "validation_f1"], ascending=False
        ).iloc[0]
    chosen_params = parameter_combinations[int(chosen["iteration"]) - 1]
    model = train_random_forest(X_train_selected, y_train, bootstrap=True, **chosen_params)
    validation_probability = model.predict_proba(X_validation_selected)[:, 1]
    threshold = best_f1_threshold(y_validation, validation_probability)
    test_probability = model.predict_proba(X_test_selected)[:, 1]
    test_metrics = evaluate_probabilities(y_test, test_probability, threshold)

    model_importance = pd.DataFrame(
        {
            "variant": variant,
            "feature": selected_names,
            "model_importance": model.feature_importances_,
        }
    ).sort_values("model_importance", ascending=False)
    result = {
        "variant": variant,
        "processed_feature_count": int(len(processed_names)),
        "selected_feature_count": int(selected_mask.sum()),
        "operating_threshold": threshold,
        "selected_tuning_iteration": int(chosen["iteration"]),
        "selected_parameters": {
            key: (None if pd.isna(chosen[key]) else chosen[key].item() if hasattr(chosen[key], "item") else chosen[key])
            for key in PARAMETER_SPACE
        },
        **test_metrics,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    print(f"[{variant}] complete: F1={result['f1']:.6f}, PR-AUC={result['pr_auc']:.6f}", flush=True)
    return result, tuning, selector_importances, model_importance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_hashes_before = directory_hashes(ARTIFACT_DIR)
    print("Building leakage-safe historical features from fraud_pipeline", flush=True)
    engineered = build_engineered_data()
    print(f"Engineered rows: {len(engineered):,}; columns: {engineered.shape[1]}", flush=True)

    variants = ["legacy_baseline", "corrected_simulation_day", "no_direct_time"]
    results = []
    tuning_frames = []
    selector_frames = []
    importance_frames = []
    for variant in variants:
        result, tuning, selector_importance, model_importance = run_variant(engineered, variant)
        results.append(result)
        tuning_frames.append(tuning)
        selector_frames.append(selector_importance)
        importance_frames.append(model_importance)
        gc.collect()

    artifact_hashes_after = directory_hashes(ARTIFACT_DIR)
    if artifact_hashes_before != artifact_hashes_after:
        raise RuntimeError("Frozen artifacts changed during the comparison experiment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(results)
    comparison.to_csv(args.output_dir / "comparison_metrics.csv", index=False)
    pd.concat(tuning_frames, ignore_index=True).to_csv(args.output_dir / "tuning_results.csv", index=False)
    pd.concat(selector_frames, ignore_index=True).to_csv(args.output_dir / "selector_importances.csv", index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(args.output_dir / "selected_model_importances.csv", index=False)
    metadata = {
        "source": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "feature_engineering_source": "fraud_pipeline.features.engineer_features",
        "split": {
            "train": [SPLIT_CONFIG.train_start, SPLIT_CONFIG.train_end],
            "validation": [SPLIT_CONFIG.validation_start, SPLIT_CONFIG.validation_end],
            "test": [SPLIT_CONFIG.test_start, SPLIT_CONFIG.test_end],
        },
        "variants": variants,
        "tuning_config": TUNING_CONFIG,
        "parameter_space": PARAMETER_SPACE,
        "frozen_artifacts_unchanged": True,
        "artifact_sha256": artifact_hashes_after,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\nFinal held-out test comparison", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"\nResults: {args.output_dir}", flush=True)
    print("Frozen artifact hashes verified unchanged.", flush=True)


if __name__ == "__main__":
    main()
