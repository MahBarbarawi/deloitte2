"""Cached loaders for the frozen fitted model artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import streamlit as st

from utils.paths import ARTIFACT_DIR


@dataclass(frozen=True)
class ModelArtifacts:
    preprocessor: object
    fraud_model: object
    confidence_model: object
    feature_mask: np.ndarray


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> ModelArtifacts:
    """Canonical lazy loader for fitted resources; used only by live inference."""
    return ModelArtifacts(
        preprocessor=joblib.load(ARTIFACT_DIR / "preprocessor.joblib"),
        fraud_model=joblib.load(ARTIFACT_DIR / "best_rf_model.joblib"),
        confidence_model=joblib.load(ARTIFACT_DIR / "confidence_model.joblib"),
        feature_mask=np.load(ARTIFACT_DIR / "rf_feature_mask.npy"),
    )


def load_preprocessor():
    return load_model_artifacts().preprocessor


def load_fraud_model():
    return load_model_artifacts().fraud_model


def load_confidence_model():
    return load_model_artifacts().confidence_model


def load_feature_mask() -> np.ndarray:
    return load_model_artifacts().feature_mask


@st.cache_data(show_spinner=False)
def load_artifact_json(filename: str):
    with (ARTIFACT_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_model_config() -> dict:
    return load_artifact_json("model_config.json")


def load_history_reference() -> dict[str, float]:
    return load_artifact_json("history_reference.json")


def load_confidence_columns() -> list[str]:
    return load_artifact_json("confidence_feature_columns.json")
