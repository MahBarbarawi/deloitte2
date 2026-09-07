"""Raw BankSim loading and source-level schema validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.paths import RAW_TRANSACTIONS_CSV


RAW_COLUMNS = (
    "step", "customer", "age", "gender", "zipcodeOri",
    "merchant", "zipMerchant", "category", "amount", "fraud",
)


@dataclass(frozen=True)
class SourceMetadata:
    path: Path
    rows: int
    columns: int
    simulation_day_min: int
    simulation_day_max: int
    unique_simulation_days: int
    fraud_transactions: int


def validate_raw_schema(data: pd.DataFrame) -> None:
    missing = [column for column in RAW_COLUMNS if column not in data.columns]
    extra = [column for column in data.columns if column not in RAW_COLUMNS]
    if missing or extra:
        raise ValueError(f"Unexpected BankSim schema; missing={missing}, extra={extra}")
    if data.columns.tolist() != list(RAW_COLUMNS):
        raise ValueError("BankSim columns are not in the canonical order.")


def source_metadata(data: pd.DataFrame, path: Path = RAW_TRANSACTIONS_CSV) -> SourceMetadata:
    validate_raw_schema(data)
    return SourceMetadata(
        path=path,
        rows=len(data),
        columns=data.shape[1],
        simulation_day_min=int(data["step"].min()),
        simulation_day_max=int(data["step"].max()),
        unique_simulation_days=int(data["step"].nunique()),
        fraud_transactions=int(pd.to_numeric(data["fraud"], errors="raise").sum()),
    )


def load_raw_transactions(path: Path = RAW_TRANSACTIONS_CSV) -> pd.DataFrame:
    data = pd.read_csv(path)
    validate_raw_schema(data)
    return data
