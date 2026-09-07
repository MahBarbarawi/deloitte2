"""Deterministic cleaning for the raw BankSim transaction table."""
from __future__ import annotations

import pandas as pd

from fraud_pipeline.data import RAW_COLUMNS, validate_raw_schema


TEXT_COLUMNS = ("customer", "age", "gender", "merchant", "category")
INTEGER_COLUMNS = ("step", "zipcodeOri", "zipMerchant", "fraud")


def clean_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Strip source quoting, enforce types, and preserve row/column order."""
    validate_raw_schema(data)
    cleaned = data.copy()

    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = (
            cleaned[column].astype(str).str.strip().str.strip("'").str.strip('"')
        )
    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].astype(str)
    for column in INTEGER_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise").astype("int64")
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="raise").astype("float64")

    if cleaned.isna().any().any():
        raise ValueError("Cleaned transactions contain missing values.")
    if not cleaned["fraud"].isin([0, 1]).all():
        raise ValueError("fraud must contain only 0 and 1.")
    if (cleaned["amount"] < 0).any():
        raise ValueError("amount cannot be negative.")
    if cleaned["step"].min() < 0:
        raise ValueError("step must be a zero-based, non-negative simulated day index.")

    return cleaned.loc[:, RAW_COLUMNS].copy()
