"""Frozen Random Forest inference with direct, model-grounded evidence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_pipeline.training import prepare_frozen_raw_inputs, transform_for_frozen_model
from utils.artifact_loader import load_model_artifacts
from utils.history_context import HistoricalContext, build_historical_context


OUTPUT_ONLY_COLUMNS = {
    "source_index", "step", "customer", "merchant", "zipcodeOri", "zipMerchant",
    "actual_fraud", "fraud", "prediction", "fraud_probability",
    "decision_margin", "tree_agreement", "tree_probability_std",
}


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    std: float
    minimum: float
    maximum: float
    q25: float
    median: float
    q75: float


@dataclass(frozen=True)
class ModelDecisionEvidence:
    fraud_probability: float
    decision_threshold: float
    decision_margin: float
    prediction: int
    tree_agreement_count: int
    tree_count: int
    tree_agreement: float
    tree_probabilities: DistributionSummary
    leaf_training_support: DistributionSummary


@dataclass(frozen=True)
class FraudAssessment:
    model: ModelDecisionEvidence
    history: HistoricalContext


def raw_model_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Adapt corrected features to the frozen input contract."""
    bundle = load_model_artifacts()
    expected = list(bundle.preprocessor.feature_names_in_)
    forbidden = set(expected) & OUTPUT_ONLY_COLUMNS
    if forbidden:
        raise ValueError(f"Saved preprocessor unexpectedly requests labels/outputs: {forbidden}")
    return prepare_frozen_raw_inputs(rows, bundle)


def _summarize(values: np.ndarray) -> DistributionSummary:
    return DistributionSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        q25=float(np.quantile(values, 0.25)),
        median=float(np.median(values)),
        q75=float(np.quantile(values, 0.75)),
    )


def assess_transactions(rows: pd.DataFrame) -> list[FraudAssessment]:
    """Score rows without retraining or consulting any secondary classifier."""
    raw = raw_model_frame(rows)
    bundle = load_model_artifacts()
    model = bundle.model
    selected = transform_for_frozen_model(raw, bundle)
    threshold = bundle.threshold

    fraud_class = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(selected)[:, fraud_class]
    predictions = probabilities >= threshold
    tree_probabilities = np.vstack([
        tree.predict_proba(selected)[:, int(np.flatnonzero(tree.classes_ == 1)[0])]
        for tree in model.estimators_
    ])
    leaf_support = np.vstack([
        tree.tree_.n_node_samples[tree.apply(selected)]
        for tree in model.estimators_
    ])

    assessments = []
    raw_for_history = rows.reset_index(drop=True)
    for index, probability in enumerate(probabilities):
        tree_votes = tree_probabilities[:, index] >= threshold
        assessments.append(FraudAssessment(
            model=ModelDecisionEvidence(
                fraud_probability=float(probability),
                decision_threshold=threshold,
                decision_margin=float(probability - threshold),
                prediction=int(predictions[index]),
                tree_agreement_count=int(np.count_nonzero(tree_votes == predictions[index])),
                tree_count=len(model.estimators_),
                tree_agreement=float(np.mean(tree_votes == predictions[index])),
                tree_probabilities=_summarize(tree_probabilities[:, index]),
                # n_node_samples is a literal unique in-bag observation count.
                # The weighted alternative mixes bootstrap multiplicity with
                # fitted class weights and is not a human-readable row count.
                leaf_training_support=_summarize(leaf_support[:, index]),
            ),
            history=build_historical_context(raw_for_history.iloc[index]),
        ))
    return assessments


def assess_transaction(row: pd.DataFrame | pd.Series) -> FraudAssessment:
    frame = row.to_frame().T if isinstance(row, pd.Series) else row
    if len(frame) != 1:
        raise ValueError("assess_transaction expects exactly one row.")
    return assess_transactions(frame)[0]


def predict_transactions(rows: pd.DataFrame) -> pd.DataFrame:
    """Compatibility tabular output containing only Random Forest evidence."""
    records = []
    for assessment in assess_transactions(rows):
        evidence = assessment.model
        records.append({
            "prediction": evidence.prediction,
            "fraud_probability": evidence.fraud_probability,
            "decision_threshold": evidence.decision_threshold,
            "decision_margin": evidence.decision_margin,
            "tree_agreement_count": evidence.tree_agreement_count,
            "tree_count": evidence.tree_count,
            "tree_agreement": evidence.tree_agreement,
            "tree_probability_mean": evidence.tree_probabilities.mean,
            "tree_probability_std": evidence.tree_probabilities.std,
            "tree_probability_min": evidence.tree_probabilities.minimum,
            "tree_probability_max": evidence.tree_probabilities.maximum,
            "leaf_support_mean": evidence.leaf_training_support.mean,
            "leaf_support_median": evidence.leaf_training_support.median,
            "leaf_support_min": evidence.leaf_training_support.minimum,
            "leaf_support_max": evidence.leaf_training_support.maximum,
        })
    return pd.DataFrame.from_records(records)
