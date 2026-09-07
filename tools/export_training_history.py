"""Rebuild Decision Evidence history from the canonical training split."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fraud_pipeline.cleaning import clean_transactions
from fraud_pipeline.data import load_raw_transactions
from fraud_pipeline.exports import export_training_history
from fraud_pipeline.splitting import chronological_split
from utils.paths import TRAINING_HISTORY_DIR


def main() -> None:
    cleaned = clean_transactions(load_raw_transactions())
    train = chronological_split(cleaned).train
    metadata = export_training_history(train)
    print(f"Training rows: {metadata['training_row_count']:,}")
    print(f"Simulation days: {metadata['training_step_min']}-{metadata['training_step_max']}")
    print(f"Split: {metadata['split_definition']}")
    print(f"Output location: {TRAINING_HISTORY_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
