"""Exact confidence feature calculations exported from the training notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_FEATURES = [
    "customer_previous_transaction_count",
    "merchant_previous_transaction_count",
    "category_previous_transaction_count",
    "customer_merchant_previous_transaction_count",
    "customer_category_previous_transaction_count",
]


def decision_margin_score(probabilities, threshold: float) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    score = np.where(
        probabilities >= threshold,
        (probabilities - threshold) / (1 - threshold),
        (threshold - probabilities) / threshold,
    )
    return np.clip(score, 0, 1)


def entropy_certainty_score(probabilities) -> np.ndarray:
    p = np.clip(np.asarray(probabilities), 1e-12, 1 - 1e-12)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return 1 - entropy


def build_confidence_features(model, x_rf, x_raw: pd.DataFrame, threshold: float, history_reference: dict):
    fraud_probability = model.predict_proba(x_rf)[:, 1]
    prediction = (fraud_probability >= threshold).astype(int)
    n_rows, n_trees = x_rf.shape[0], len(model.estimators_)
    agreement_count = np.zeros(n_rows, dtype=np.int32)
    probability_sum = np.zeros(n_rows, dtype=np.float64)
    probability_squared_sum = np.zeros(n_rows, dtype=np.float64)

    for tree in model.estimators_:
        tree_probability = tree.predict_proba(x_rf)[:, 1]
        tree_prediction = (tree_probability >= threshold).astype(int)
        agreement_count += tree_prediction == prediction
        probability_sum += tree_probability
        probability_squared_sum += tree_probability**2

    tree_mean = probability_sum / n_trees
    tree_variance = np.maximum(probability_squared_sum / n_trees - tree_mean**2, 0)
    features = pd.DataFrame(
        {
            "fraud_probability": fraud_probability,
            "predicted_fraud": prediction,
            "decision_margin": decision_margin_score(fraud_probability, threshold),
            "entropy_certainty": entropy_certainty_score(fraud_probability),
            "tree_agreement": agreement_count / n_trees,
            "tree_probability_std": np.sqrt(tree_variance),
        }
    )
    for feature in HISTORY_FEATURES:
        values = x_raw[feature].to_numpy()
        reference = history_reference[feature]
        features[f"support__{feature}"] = np.clip(
            np.log1p(values) / np.log1p(reference), 0, 1
        )
    return features, prediction, fraud_probability


def confidence_label(value: float, config: dict) -> str:
    thresholds = config["confidence_config"]
    if value >= thresholds["very_high_confidence"]:
        return "VERY HIGH"
    if value >= thresholds["high_confidence"]:
        return "HIGH"
    if value >= thresholds["medium_confidence"]:
        return "MEDIUM"
    return "LOW"
