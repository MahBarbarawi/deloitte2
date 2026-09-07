"""Chronological BankSim split logic—the sole owner of split boundaries."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_pipeline.config import SPLIT_CONFIG, SplitConfig


@dataclass(frozen=True)
class ChronologicalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(data: pd.DataFrame, config: SplitConfig = SPLIT_CONFIG) -> ChronologicalSplits:
    config.validate()
    if "step" not in data or "fraud" not in data:
        raise ValueError("Chronological splitting requires step and fraud columns.")
    observed_min = int(data["step"].min())
    observed_max = int(data["step"].max())
    if observed_min != config.train_start or observed_max != config.test_end:
        raise ValueError(
            f"Dataset simulation-day range {observed_min}-{observed_max} does not match {config.rule}."
        )

    splits = ChronologicalSplits(
        train=data.loc[data["step"].between(config.train_start, config.train_end)].copy(),
        validation=data.loc[data["step"].between(config.validation_start, config.validation_end)].copy(),
        test=data.loc[data["step"].between(config.test_start, config.test_end)].copy(),
    )
    if sum(map(len, (splits.train, splits.validation, splits.test))) != len(data):
        raise RuntimeError("Chronological split did not assign every row exactly once.")
    if not (
        splits.train["step"].max() < splits.validation["step"].min()
        and splits.validation["step"].max() < splits.test["step"].min()
    ):
        raise RuntimeError("Chronological split overlap detected.")
    return splits


def summarize_splits(splits: ChronologicalSplits) -> pd.DataFrame:
    rows = []
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        fraud = int(frame["fraud"].sum())
        rows.append({
            "split": name,
            "simulation_day_min": int(frame["step"].min()),
            "simulation_day_max": int(frame["step"].max()),
            "rows": len(frame),
            "fraud": fraud,
            "legitimate": len(frame) - fraud,
            "fraud_rate": float(frame["fraud"].mean()),
        })
    return pd.DataFrame(rows)
