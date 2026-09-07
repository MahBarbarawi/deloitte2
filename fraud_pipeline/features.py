"""Leakage-safe historical and behavioral feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_pipeline.history import (
    ENTITY_AMOUNT_RENAMES,
    build_entity_history_features,
    build_relationship_history_features,
)


MODEL_DROP_COLUMNS = (
    "zipcodeOri",
    "zipMerchant",
    "customer",
    "merchant",
    "merchant_steps_since_last_transaction",
    "category_steps_since_last_transaction",
)


def engineer_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build model features using only observations from earlier simulated days."""
    source = cleaned.copy().reset_index(drop=True)
    source["_row_id"] = np.arange(len(source))
    for column in ("customer", "merchant", "category", "age", "gender"):
        source[column] = source[column].astype(str).str.strip().str.strip("'").str.strip('"')
    for column in ("step", "amount", "fraud"):
        source[column] = pd.to_numeric(source[column], errors="raise")

    # BankSim exposes only a simulated-day index. No hour or intraday feature is derived.
    source["log_amount"] = np.log1p(source["amount"])
    customer = build_entity_history_features(source, "customer", "customer")
    merchant = build_entity_history_features(source, "merchant", "merchant")
    category = build_entity_history_features(source, "category", "category")
    engineered = (
        source.merge(customer, on=["customer", "step"], how="left", validate="many_to_one")
        .merge(merchant, on=["merchant", "step"], how="left", validate="many_to_one")
        .merge(category, on=["category", "step"], how="left", validate="many_to_one")
    )
    engineered["customer_amount_ratio"] = (
        engineered["amount"] / engineered["customer_previous_avg_amount"].replace(0, np.nan)
    )
    engineered["merchant_amount_ratio"] = (
        engineered["amount"] / engineered["merchant_previous_avg_amount"].replace(0, np.nan)
    )
    engineered["category_amount_ratio"] = (
        engineered["amount"] / engineered["category_previous_avg_amount"].replace(0, np.nan)
    )
    engineered = engineered.rename(columns=ENTITY_AMOUNT_RENAMES).drop(columns=["is_new_category"], errors="ignore")

    customer_merchant = build_relationship_history_features(
        source, ["customer", "merchant"], "customer_merchant"
    )
    customer_category = build_relationship_history_features(
        source, ["customer", "category"], "customer_category"
    )
    engineered = (
        engineered.merge(
            customer_merchant,
            on=["customer", "merchant", "step"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            customer_category,
            on=["customer", "category", "step"],
            how="left",
            validate="many_to_one",
        )
    )
    engineered["customer_merchant_amount_ratio"] = (
        engineered["amount"] / engineered["customer_merchant_previous_avg_amount"].replace(0, np.nan)
    )
    engineered["customer_category_amount_ratio"] = (
        engineered["amount"] / engineered["customer_category_previous_avg_amount"].replace(0, np.nan)
    )
    engineered["customer_merchant_transaction_share"] = (
        engineered["customer_merchant_previous_transaction_count"]
        / engineered["customer_previous_transaction_count"].replace(0, np.nan)
    )
    engineered["customer_merchant_spend_share"] = (
        engineered["customer_merchant_previous_total_amount"]
        / engineered["customer_previous_total_spend"].replace(0, np.nan)
    )
    engineered["customer_category_transaction_share"] = (
        engineered["customer_category_previous_transaction_count"]
        / engineered["customer_previous_transaction_count"].replace(0, np.nan)
    )
    engineered["customer_category_spend_share"] = (
        engineered["customer_category_previous_total_amount"]
        / engineered["customer_previous_total_spend"].replace(0, np.nan)
    )
    engineered = engineered.rename(columns={
        "is_new_customer_merchant": "is_new_merchant_for_customer",
        "is_new_customer_category": "is_new_category_for_customer",
    })
    engineered = engineered.sort_values("_row_id").drop(columns="_row_id").reset_index(drop=True)

    # The notebook's chosen structural missing strategy was zero. Every missing
    # value here denotes unavailable prior history or a ratio with no baseline.
    numeric_missing = engineered.columns[engineered.isna().any()].tolist()
    non_numeric_missing = [
        column for column in numeric_missing
        if not pd.api.types.is_numeric_dtype(engineered[column])
    ]
    if non_numeric_missing:
        raise ValueError(f"Unexpected missing categorical values: {non_numeric_missing}")
    engineered[numeric_missing] = engineered[numeric_missing].fillna(0)
    return engineered


def model_dataset(engineered: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers/redundant fields while retaining step for splitting."""
    return engineered.drop(columns=list(MODEL_DROP_COLUMNS), errors="ignore").copy()
