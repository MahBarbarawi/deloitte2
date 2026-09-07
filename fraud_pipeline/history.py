"""Shared historical definitions for model features and dashboard context.

For a transaction on simulated day T, model-history features use only rows from
simulated days strictly less than T. Because BankSim has no within-day ordering,
transactions on the same simulated day never contribute to one another.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ENTITY_AMOUNT_RENAMES = {
    "customer_previous_total_amount": "customer_previous_total_spend",
    "customer_previous_avg_amount": "customer_previous_avg_spend",
    "customer_previous_std_amount": "customer_previous_std_spend",
    "customer_previous_min_amount": "customer_previous_min_spend",
    "customer_previous_max_amount": "customer_previous_max_spend",
    "customer_amount_per_step": "customer_spend_per_step",
    "merchant_previous_total_amount": "merchant_previous_total_received",
    "merchant_amount_per_step": "merchant_received_per_step",
}


def build_entity_history_features(
    source: pd.DataFrame,
    entity_col: str,
    prefix: str,
    recent_windows: tuple[int, ...] = (1, 3, 6, 24),
) -> pd.DataFrame:
    """Build day-isolated historical features for one entity dimension."""
    temp = source[[entity_col, "step", "amount"]].copy()
    temp["_amount_squared"] = temp["amount"] ** 2
    stats = (
        temp.groupby([entity_col, "step"], observed=True, sort=True)
        .agg(
            step_transaction_count=("amount", "size"),
            step_total_amount=("amount", "sum"),
            step_amount_squared_sum=("_amount_squared", "sum"),
            step_min_amount=("amount", "min"),
            step_max_amount=("amount", "max"),
        )
        .reset_index()
        .sort_values([entity_col, "step"])
        .reset_index(drop=True)
    )
    grouped = stats.groupby(entity_col, sort=False)
    previous_count = grouped["step_transaction_count"].cumsum() - stats["step_transaction_count"]
    previous_total = grouped["step_total_amount"].cumsum() - stats["step_total_amount"]
    previous_squared = grouped["step_amount_squared_sum"].cumsum() - stats["step_amount_squared_sum"]
    safe_count = previous_count.replace(0, np.nan)

    stats[f"{prefix}_previous_transaction_count"] = previous_count
    stats[f"{prefix}_previous_total_amount"] = previous_total
    stats[f"{prefix}_previous_avg_amount"] = previous_total / safe_count
    variance_numerator = (previous_squared - previous_total**2 / safe_count).clip(lower=0)
    stats[f"{prefix}_previous_std_amount"] = np.sqrt(
        variance_numerator / (previous_count - 1).where(previous_count > 1)
    )
    stats["_shifted_min"] = grouped["step_min_amount"].shift(1)
    stats["_shifted_max"] = grouped["step_max_amount"].shift(1)
    stats[f"{prefix}_previous_min_amount"] = stats.groupby(entity_col, sort=False)["_shifted_min"].cummin()
    stats[f"{prefix}_previous_max_amount"] = stats.groupby(entity_col, sort=False)["_shifted_max"].cummax()
    stats[f"{prefix}_steps_since_last_transaction"] = stats["step"] - grouped["step"].shift(1)

    first_step = grouped["step"].transform("min")
    elapsed = (stats["step"] - first_step).astype(float)
    stats[f"{prefix}_transactions_per_step"] = previous_count / elapsed.where(elapsed > 0)
    stats[f"{prefix}_amount_per_step"] = previous_total / elapsed.where(elapsed > 0)

    all_steps = np.arange(int(source["step"].min()), int(source["step"].max()) + 1)
    count_matrix = (
        stats.pivot(index=entity_col, columns="step", values="step_transaction_count")
        .reindex(columns=all_steps, fill_value=0)
        .fillna(0)
    )
    amount_matrix = (
        stats.pivot(index=entity_col, columns="step", values="step_total_amount")
        .reindex(columns=all_steps, fill_value=0)
        .fillna(0)
    )
    shifted_counts = count_matrix.shift(1, axis=1).fillna(0)
    shifted_amounts = amount_matrix.shift(1, axis=1).fillna(0)
    recent = []
    for window in recent_windows:
        recent.extend([
            shifted_counts.T.rolling(window=window, min_periods=1).sum().T.stack(
                future_stack=True
            ).rename(f"{prefix}_transactions_last_{window}_steps"),
            shifted_amounts.T.rolling(window=window, min_periods=1).sum().T.stack(
                future_stack=True
            ).rename(f"{prefix}_amount_last_{window}_steps"),
        ])
    recent_features = pd.concat(recent, axis=1).reset_index()

    generated = [column for column in stats if column.startswith(prefix + "_")]
    features = stats[[entity_col, "step", *generated]].merge(
        recent_features, on=[entity_col, "step"], how="left", validate="one_to_one"
    )
    features[f"{prefix}_velocity_ratio"] = (
        features[f"{prefix}_transactions_last_1_steps"]
        / features[f"{prefix}_transactions_per_step"].replace(0, np.nan)
    )
    features[f"{prefix}_recent_amount_ratio"] = (
        features[f"{prefix}_amount_last_1_steps"]
        / features[f"{prefix}_amount_per_step"].replace(0, np.nan)
    )
    features[f"is_new_{prefix}"] = (features[f"{prefix}_previous_transaction_count"] == 0).astype(int)
    return features


def build_relationship_history_features(
    source: pd.DataFrame,
    relationship_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Build day-isolated history for a customer/entity relationship."""
    temp = source[[*relationship_cols, "step", "amount"]].copy()
    temp["_amount_squared"] = temp["amount"] ** 2
    stats = (
        temp.groupby([*relationship_cols, "step"], observed=True, sort=True)
        .agg(
            step_transaction_count=("amount", "size"),
            step_total_amount=("amount", "sum"),
            step_amount_squared_sum=("_amount_squared", "sum"),
            step_min_amount=("amount", "min"),
            step_max_amount=("amount", "max"),
        )
        .reset_index()
        .sort_values([*relationship_cols, "step"])
        .reset_index(drop=True)
    )
    grouped = stats.groupby(relationship_cols, sort=False)
    previous_count = grouped["step_transaction_count"].cumsum() - stats["step_transaction_count"]
    previous_total = grouped["step_total_amount"].cumsum() - stats["step_total_amount"]
    previous_squared = grouped["step_amount_squared_sum"].cumsum() - stats["step_amount_squared_sum"]
    safe_count = previous_count.replace(0, np.nan)
    stats[f"{prefix}_previous_transaction_count"] = previous_count
    stats[f"{prefix}_previous_total_amount"] = previous_total
    stats[f"{prefix}_previous_avg_amount"] = previous_total / safe_count
    variance_numerator = (previous_squared - previous_total**2 / safe_count).clip(lower=0)
    stats[f"{prefix}_previous_std_amount"] = np.sqrt(
        variance_numerator / (previous_count - 1).where(previous_count > 1)
    )
    stats["_shifted_min"] = grouped["step_min_amount"].shift(1)
    stats["_shifted_max"] = grouped["step_max_amount"].shift(1)
    stats[f"{prefix}_previous_min_amount"] = stats.groupby(relationship_cols, sort=False)["_shifted_min"].cummin()
    stats[f"{prefix}_previous_max_amount"] = stats.groupby(relationship_cols, sort=False)["_shifted_max"].cummax()
    stats[f"{prefix}_steps_since_last_transaction"] = stats["step"] - grouped["step"].shift(1)
    stats[f"is_new_{prefix}"] = (previous_count == 0).astype(int)
    generated = [
        column for column in stats
        if column.startswith(prefix + "_") or column == f"is_new_{prefix}"
    ]
    return stats[[*relationship_cols, "step", *generated]]


def _entity_summary(data: pd.DataFrame, entity: str) -> pd.DataFrame:
    return (
        data.groupby(entity, observed=True, sort=True)["amount"]
        .agg(transaction_count="size", amount_mean="mean", amount_min="min", amount_max="max")
        .reset_index()
    )


def _relationship_summary(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return data.groupby(keys, observed=True, sort=True).size().rename("transaction_count").reset_index()


def build_training_history_reference(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the minimum label-free history required by Decision Evidence."""
    required = {"step", "customer", "merchant", "category", "amount"}
    if missing := sorted(required - set(train.columns)):
        raise ValueError(f"Training history source is missing columns: {missing}")
    return {
        "customers.parquet": _entity_summary(train, "customer"),
        "merchants.parquet": _entity_summary(train, "merchant"),
        "categories.parquet": _entity_summary(train, "category"),
        "customer_merchants.parquet": _relationship_summary(train, ["customer", "merchant"]),
        "customer_categories.parquet": _relationship_summary(train, ["customer", "category"]),
    }


@dataclass(frozen=True)
class LoadedHistoryReference:
    customers: pd.DataFrame
    merchants: pd.DataFrame
    categories: pd.DataFrame
    customer_merchants: pd.Series
    customer_categories: pd.Series
    metadata: dict


def load_history_reference(directory: Path) -> LoadedHistoryReference:
    import json

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    def entity_table(name: str, key: str) -> pd.DataFrame:
        frame = pd.read_parquet(directory / name)
        frame[key] = frame[key].astype(str)
        return frame.set_index(key)

    def relationship_table(name: str, keys: list[str]) -> pd.Series:
        frame = pd.read_parquet(directory / name)
        for key in keys:
            frame[key] = frame[key].astype(str)
        return frame.set_index(keys)["transaction_count"]

    return LoadedHistoryReference(
        customers=entity_table("customers.parquet", "customer"),
        merchants=entity_table("merchants.parquet", "merchant"),
        categories=entity_table("categories.parquet", "category"),
        customer_merchants=relationship_table("customer_merchants.parquet", ["customer", "merchant"]),
        customer_categories=relationship_table("customer_categories.parquet", ["customer", "category"]),
        metadata=metadata,
    )


@dataclass(frozen=True)
class EntityHistory:
    transaction_count: int
    amount_mean: float | None
    amount_min: float | None
    amount_max: float | None

    @property
    def is_new(self) -> bool:
        return self.transaction_count == 0


@dataclass(frozen=True)
class EntityFamiliarity:
    customer: EntityHistory
    merchant: EntityHistory
    category: EntityHistory
    customer_merchant_count: int
    customer_category_count: int


@dataclass(frozen=True)
class BehaviorFamiliarity:
    customer_category_frequency: float | None
    customer_amount_ratio: float | None
    merchant_amount_ratio: float | None
    category_amount_ratio: float | None


@dataclass(frozen=True)
class SimilarHistory:
    available: bool
    reason: str


@dataclass(frozen=True)
class HistoricalContext:
    entity: EntityFamiliarity
    behavior: BehaviorFamiliarity
    similar: SimilarHistory
    training_row_count: int
    training_step_min: int
    training_step_max: int
    category_label: str


def _entity(table: pd.DataFrame, key: str) -> EntityHistory:
    if key not in table.index:
        return EntityHistory(0, None, None, None)
    row = table.loc[key]
    return EntityHistory(
        int(row["transaction_count"]),
        float(row["amount_mean"]),
        float(row["amount_min"]),
        float(row["amount_max"]),
    )


def _relationship_count(table: pd.Series, keys: tuple[str, str]) -> int:
    try:
        return int(table.loc[keys])
    except KeyError:
        return 0


def _ratio(amount: float, mean: float | None) -> float | None:
    return amount / mean if mean is not None and mean > 0 else None


def get_historical_context(reference: LoadedHistoryReference, row: pd.Series) -> HistoricalContext:
    """Query factual entity/behavior history from a prepared training reference."""
    customer = str(row["customer"])
    merchant = str(row["merchant"])
    category = str(row["category"])
    amount = float(row["amount"])
    customer_history = _entity(reference.customers, customer)
    merchant_history = _entity(reference.merchants, merchant)
    category_history = _entity(reference.categories, category)
    customer_merchant_count = _relationship_count(reference.customer_merchants, (customer, merchant))
    customer_category_count = _relationship_count(reference.customer_categories, (customer, category))
    return HistoricalContext(
        entity=EntityFamiliarity(
            customer_history,
            merchant_history,
            category_history,
            customer_merchant_count,
            customer_category_count,
        ),
        behavior=BehaviorFamiliarity(
            customer_category_frequency=(
                customer_category_count / customer_history.transaction_count
                if customer_history.transaction_count else None
            ),
            customer_amount_ratio=_ratio(amount, customer_history.amount_mean),
            merchant_amount_ratio=_ratio(amount, merchant_history.amount_mean),
            category_amount_ratio=_ratio(amount, category_history.amount_mean),
        ),
        similar=SimilarHistory(
            available=False,
            reason=(
                "No similar-history result is shown because the frozen project does not define a "
                "validated similarity metric or neighborhood. Inventing one would make the evidence arbitrary."
            ),
        ),
        training_row_count=int(reference.metadata["training_row_count"]),
        training_step_min=int(reference.metadata["training_step_min"]),
        training_step_max=int(reference.metadata["training_step_max"]),
        category_label=category.replace("es_", "").replace("_", " ").title(),
    )
