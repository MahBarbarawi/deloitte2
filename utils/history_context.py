"""Cached runtime adapter for canonical pipeline history functions."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from fraud_pipeline.config import HISTORY_SCHEMA_VERSION, SPLIT_CONFIG
from fraud_pipeline.history import (
    HistoricalContext,
    LoadedHistoryReference,
    get_historical_context,
    load_history_reference,
)
from utils.paths import TRAINING_HISTORY_DIR


@lru_cache(maxsize=1)
def load_training_history() -> LoadedHistoryReference:
    reference = load_history_reference(TRAINING_HISTORY_DIR)
    metadata = reference.metadata
    expected = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "split_definition": SPLIT_CONFIG.rule,
        "training_step_min": SPLIT_CONFIG.train_start,
        "training_step_max": SPLIT_CONFIG.train_end,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Training-history metadata is incompatible: {mismatches}")
    return reference


def build_historical_context(row: pd.Series) -> HistoricalContext:
    return get_historical_context(load_training_history(), row)
