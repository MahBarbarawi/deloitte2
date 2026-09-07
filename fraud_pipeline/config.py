"""Canonical, versioned configuration for the BankSim pipeline."""
from __future__ import annotations

from dataclasses import dataclass


PIPELINE_VERSION = "2.0.0"
HISTORY_SCHEMA_VERSION = 2
DASHBOARD_SCHEMA_VERSION = 2
FROZEN_DECISION_THRESHOLD = 0.7935502066058817


@dataclass(frozen=True)
class SplitConfig:
    """Inclusive chronological boundaries used by the frozen model project."""

    train_start: int = 0
    train_end: int = 125
    validation_start: int = 126
    validation_end: int = 152
    test_start: int = 153
    test_end: int = 179

    def validate(self) -> None:
        if not (
            self.train_start <= self.train_end
            < self.validation_start <= self.validation_end
            < self.test_start <= self.test_end
        ):
            raise ValueError("Chronological split ranges must be ordered and non-overlapping.")
        if self.validation_start != self.train_end + 1 or self.test_start != self.validation_end + 1:
            raise ValueError("Chronological split ranges must be contiguous.")

    @property
    def rule(self) -> str:
        return (
            f"train={self.train_start}-{self.train_end}; "
            f"validation={self.validation_start}-{self.validation_end}; "
            f"test={self.test_start}-{self.test_end}"
        )


SPLIT_CONFIG = SplitConfig()
SPLIT_CONFIG.validate()
