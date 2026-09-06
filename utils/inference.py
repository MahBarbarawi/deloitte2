"""Frozen preprocessing, fraud scoring, thresholding, and confidence inference."""
from __future__ import annotations

import pandas as pd

from utils.artifact_loader import (
    load_confidence_columns,
    load_confidence_model,
    load_feature_mask,
    load_fraud_model,
    load_history_reference,
    load_model_config,
    load_preprocessor,
)
from utils.confidence import build_confidence_features, confidence_label

OUTPUT_ONLY_COLUMNS = {
    "source_index", "step", "customer", "merchant", "zipcodeOri", "zipMerchant",
    "actual_fraud", "fraud", "prediction", "fraud_probability",
    "prediction_confidence", "prediction_correct", "predicted_fraud",
    "decision_margin", "entropy_certainty", "tree_agreement",
    "tree_probability_std", "confidence_level",
}


def raw_model_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Select and order only the raw columns the saved preprocessor expects."""
    preprocessor = load_preprocessor()
    expected = list(preprocessor.feature_names_in_)
    missing = [column for column in expected if column not in rows.columns]
    if missing:
        raise ValueError(f"Missing model inputs: {', '.join(missing)}")
    forbidden = set(expected) & OUTPUT_ONLY_COLUMNS
    if forbidden:
        raise ValueError(f"Saved preprocessor unexpectedly requests labels/outputs: {forbidden}")
    return rows.loc[:, expected].copy()


def predict_transactions(rows: pd.DataFrame) -> pd.DataFrame:
    raw = raw_model_frame(rows)
    preprocessor = load_preprocessor()
    model = load_fraud_model()
    mask = load_feature_mask()
    config = load_model_config()
    threshold = float(config["decision_threshold"])

    processed = preprocessor.transform(raw)
    selected = processed[:, mask]
    confidence_features, predictions, probabilities = build_confidence_features(
        model, selected, raw.reset_index(drop=True), threshold, load_history_reference()
    )
    columns = load_confidence_columns()
    confidence_features = confidence_features.loc[:, columns]
    confidence = load_confidence_model().predict_proba(confidence_features)[:, 1]
    output = confidence_features.copy()
    output["prediction"] = predictions
    output["fraud_probability"] = probabilities
    output["prediction_confidence"] = confidence
    output["confidence_level"] = [confidence_label(value, config) for value in confidence]
    return output
