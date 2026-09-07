"""Cached dashboard adapters for canonical frozen pipeline artifacts."""
from __future__ import annotations

import json

import streamlit as st

from fraud_pipeline.training import FrozenModelBundle, load_frozen_model_bundle
from utils.paths import ARTIFACT_DIR


ModelArtifacts = FrozenModelBundle


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> ModelArtifacts:
    return load_frozen_model_bundle()


def load_preprocessor():
    return load_model_artifacts().preprocessor


def load_fraud_model():
    return load_model_artifacts().model


def load_feature_mask():
    return load_model_artifacts().feature_mask


@st.cache_data(show_spinner=False)
def load_artifact_json(filename: str):
    with (ARTIFACT_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_model_config() -> dict:
    return load_artifact_json("model_config.json")
