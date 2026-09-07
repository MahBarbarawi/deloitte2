"""Preprocessing, selection, and explicit model-training APIs.

The compatibility artifact build loads the frozen model through
``load_frozen_model_bundle``. It never calls ``train_random_forest``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_pipeline.config import FROZEN_DECISION_THRESHOLD
from utils.paths import (
    FROZEN_MODEL_CONFIG,
    FROZEN_PREPROCESSOR,
    FROZEN_RANDOM_FOREST,
    SELECTED_FEATURE_NAMES,
)


CATEGORICAL_FEATURES = ("age", "gender", "category")
LEGACY_EXCLUDED_TEMPORAL_INPUTS = ("day", "hour_of_day")


@dataclass(frozen=True)
class FrozenModelBundle:
    preprocessor: object
    model: RandomForestClassifier
    feature_mask: np.ndarray
    selected_feature_names: tuple[str, ...]
    threshold: float


def load_frozen_model_bundle() -> FrozenModelBundle:
    preprocessor = joblib.load(FROZEN_PREPROCESSOR)
    model = joblib.load(FROZEN_RANDOM_FOREST)
    selected = tuple(json.loads(SELECTED_FEATURE_NAMES.read_text(encoding="utf-8")))
    processed = tuple(preprocessor.get_feature_names_out())
    selected_set = set(selected)
    mask = np.asarray([name in selected_set for name in processed], dtype=bool)
    ordered = tuple(name for name, keep in zip(processed, mask) if keep)
    if ordered != selected or int(mask.sum()) != model.n_features_in_:
        raise RuntimeError("Frozen selected-feature names do not align with the fitted model.")
    if any(f"numeric__{name}" in selected_set for name in LEGACY_EXCLUDED_TEMPORAL_INPUTS):
        raise RuntimeError("Frozen model unexpectedly depends on invalid temporal inputs.")
    config = json.loads(FROZEN_MODEL_CONFIG.read_text(encoding="utf-8"))
    threshold = float(config["decision_threshold"])
    if threshold != FROZEN_DECISION_THRESHOLD:
        raise RuntimeError("Frozen decision threshold differs from canonical compatibility value.")
    return FrozenModelBundle(preprocessor, model, mask, selected, threshold)


def prepare_frozen_raw_inputs(features: pd.DataFrame, bundle: FrozenModelBundle) -> pd.DataFrame:
    """Adapt corrected features to an old preprocessor without deriving false time fields.

    The fitted preprocessor still declares ``day`` and ``hour_of_day`` inputs.
    Both are absent from corrected pipeline data and excluded by the frozen RF
    selection mask. Zero placeholders satisfy the old transformer contract; the
    corresponding transformed columns are discarded before model inference.
    """
    frame = features.copy()
    expected = list(bundle.preprocessor.feature_names_in_)
    for column in LEGACY_EXCLUDED_TEMPORAL_INPUTS:
        if column in expected and column not in frame:
            frame[column] = 0.0
    missing = [column for column in expected if column not in frame]
    if missing:
        raise ValueError(f"Missing frozen model inputs: {missing}")
    return frame.loc[:, expected]


def transform_for_frozen_model(features: pd.DataFrame, bundle: FrozenModelBundle):
    raw = prepare_frozen_raw_inputs(features, bundle)
    return bundle.preprocessor.transform(raw)[:, bundle.feature_mask]


def score_with_frozen_model(features: pd.DataFrame, bundle: FrozenModelBundle) -> tuple[np.ndarray, np.ndarray]:
    selected = transform_for_frozen_model(features, bundle)
    fraud_class = int(np.flatnonzero(bundle.model.classes_ == 1)[0])
    probabilities = bundle.model.predict_proba(selected)[:, fraud_class]
    return probabilities, (probabilities >= bundle.threshold).astype("int8")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build the corrected preprocessing methodology for a future retrain."""
    numeric = [column for column in features if column not in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
    ])


def select_random_forest_features(processed, target: pd.Series) -> SelectFromModel:
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        threshold="median",
    )
    return selector.fit(processed, target)


def train_random_forest(selected, target: pd.Series, **parameters) -> RandomForestClassifier:
    """Explicit retraining API; never invoked by compatibility builds."""
    defaults = {
        "n_estimators": 200,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    defaults.update(parameters)
    return RandomForestClassifier(**defaults).fit(selected, target)
