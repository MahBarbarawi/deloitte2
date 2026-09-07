"""Export the Fraud Analysis parquet source as a human-readable CSV."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.paths import DASHBOARD_CSV, DASHBOARD_PARQUET

PARQUET_PATH = DASHBOARD_PARQUET
CSV_PATH = DASHBOARD_CSV


def main() -> None:
    data = pd.read_parquet(PARQUET_PATH)
    data.to_csv(CSV_PATH, index=False)

    # Read the export back with the parquet's string columns preserved so the
    # validation compares every row and value, not only summary statistics.
    string_columns = {
        column: "object"
        for column in data.columns
        if data[column].dtype == "object"
    }
    exported = pd.read_csv(
        CSV_PATH,
        dtype=string_columns,
        float_precision="round_trip",
        keep_default_na=False,
    )
    pd.testing.assert_frame_equal(data, exported, check_dtype=True, check_exact=True)

    assert len(data) == 594_643
    assert int(data["fraud"].sum()) == 7_200
    assert int(data["step"].nunique()) == 180
    assert (int(data["step"].min()), int(data["step"].max())) == (0, 179)
    assert len(exported) == len(data)
    assert list(exported.columns) == list(data.columns)
    assert int(exported["fraud"].sum()) == int(data["fraud"].sum())
    assert exported["amount"].min() == data["amount"].min()
    assert exported["amount"].max() == data["amount"].max()

    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"Column names: {list(data.columns)}")
    print(f"Output: {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print("Validation: CSV rows, values, dtypes, columns, order, and source-level totals match parquet.")


if __name__ == "__main__":
    main()
