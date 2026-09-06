"""Cached dataset and exported-result loaders."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from utils.paths import DASHBOARD_PARQUET, RESULTS_DIR, TRANSACTIONS_CSV, VALIDATION_DEMO_PARQUET

DASHBOARD_ANALYSIS_COLUMNS = ("step", "customer", "age", "gender", "merchant", "category", "amount", "fraud")
DASHBOARD_CATEGORY_COLUMNS = ("customer", "age", "gender", "merchant", "category")


@st.cache_data(show_spinner=False)
def load_dashboard_data(columns: tuple[str, ...] = DASHBOARD_ANALYSIS_COLUMNS) -> pd.DataFrame:
    """Prefer the compact exported parquet, falling back to the source CSV."""
    path = DASHBOARD_PARQUET if DASHBOARD_PARQUET.exists() else TRANSACTIONS_CSV
    if path.suffix == ".parquet":
        data = pd.read_parquet(path, columns=list(columns))
    else:
        data = pd.read_csv(path, usecols=list(columns))
    for column in DASHBOARD_CATEGORY_COLUMNS:
        if column in data.columns:
            data[column] = data[column].astype("category")
    return data


@st.cache_data(show_spinner=False)
def load_dashboard_summary() -> dict[str, float]:
    """Load only the target column needed by the executive overview."""
    fraud = load_dashboard_data(("fraud",))["fraud"]
    transactions = int(len(fraud))
    fraud_count = int(fraud.sum())
    return {
        "transactions": transactions,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_count / transactions if transactions else 0.0,
    }


@st.cache_data(show_spinner=False, max_entries=128)
def load_validation_demo(columns: tuple[str, ...] | None = None, source_index: int | None = None) -> pd.DataFrame:
    filters = [("source_index", "=", int(source_index))] if source_index is not None else None
    return pd.read_parquet(
        VALIDATION_DEMO_PARQUET,
        columns=list(columns) if columns else None,
        filters=filters,
    )


VALIDATION_INDEX_COLUMNS = [
    "source_index", "step", "customer", "merchant", "category", "amount", "age", "gender",
    "day", "hour_of_day", "actual_fraud", "prediction", "fraud_probability", "prediction_confidence",
]


@st.cache_data(show_spinner=False)
def load_validation_index() -> pd.DataFrame:
    """Compact browsing index; avoids loading 111 columns for transaction selection."""
    data = load_validation_demo(tuple(VALIDATION_INDEX_COLUMNS))
    for column in ("customer", "merchant", "category", "age", "gender"):
        data[column] = data[column].astype("category")
    return data


@st.cache_data(show_spinner=False)
def load_validation_filter_options() -> dict:
    data = load_validation_index()
    return {
        "categories": tuple(sorted(data["category"].dropna().unique())),
        "amount_min": float(data["amount"].min()),
        "amount_max": float(data["amount"].max()),
    }


@st.cache_data(show_spinner=False, max_entries=128)
def load_validation_transaction(source_index: int) -> pd.DataFrame:
    """Load one complete transaction row using parquet predicate pushdown."""
    return load_validation_demo(source_index=int(source_index)).head(1)


@st.cache_data(show_spinner=False)
def load_filter_options() -> dict:
    data = load_dashboard_data(("step", "amount", "age", "gender", "category"))
    return {
        "step_min": int(data["step"].min()), "step_max": int(data["step"].max()),
        "amount_min": float(data["amount"].min()), "amount_max": float(data["amount"].max()),
        "ages": tuple(sorted(data["age"].astype(str).dropna().unique())),
        "genders": tuple(sorted(data["gender"].astype(str).dropna().unique())),
        "categories": tuple(sorted(data["category"].dropna().unique())),
    }


@st.cache_data(show_spinner=False)
def load_result_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / filename)


@st.cache_data(show_spinner=False)
def load_result_json(filename: str) -> dict:
    with (RESULTS_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_confidence_quality() -> dict[str, float]:
    """Read only two columns rather than the full 16-column validation export."""
    from sklearn.metrics import roc_auc_score

    results = pd.read_csv(
        RESULTS_DIR / "confidence_results.csv",
        usecols=["prediction_confidence", "prediction_correct"],
    )
    return {
        "brier": float(((results.prediction_confidence - results.prediction_correct) ** 2).mean()),
        "correctness_auc": float(roc_auc_score(results.prediction_correct, results.prediction_confidence)),
    }


def existing_paths(paths: list[Path]) -> dict[str, bool]:
    return {str(path): path.exists() for path in paths}
