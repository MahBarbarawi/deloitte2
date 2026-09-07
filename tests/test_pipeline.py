from __future__ import annotations

import io
import json
import unittest
import zipfile

import numpy as np
import pandas as pd

from fraud_pipeline.cleaning import clean_transactions
from fraud_pipeline.config import FROZEN_DECISION_THRESHOLD, SPLIT_CONFIG
from fraud_pipeline.data import RAW_COLUMNS, load_raw_transactions
from fraud_pipeline.features import engineer_features
from fraud_pipeline.history import build_training_history_reference
from fraud_pipeline.splitting import chronological_split, summarize_splits
from utils.data_loader import load_dashboard_data, load_validation_demo
from utils.paths import (
    BUNDLE_ARCHIVE,
    DASHBOARD_PARQUET,
    PIPELINE_MANIFEST,
    PROCESSED_TRANSACTIONS_PARQUET,
    PROJECT_ROOT,
    TEST_SPLIT_PARQUET,
    TRAINING_HISTORY_DIR,
    TRAIN_SPLIT_PARQUET,
    VALIDATION_SPLIT_PARQUET,
)


class PipelineTests(unittest.TestCase):
    def test_cleaning_is_deterministic_and_semantically_correct(self):
        raw = load_raw_transactions().head(100)
        first = clean_transactions(raw)
        second = clean_transactions(raw)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        self.assertEqual(first.columns.tolist(), list(RAW_COLUMNS))
        self.assertFalse(first.select_dtypes("object").apply(lambda col: col.str.contains("'").any()).any())

    def test_canonical_chronological_split(self):
        cleaned = clean_transactions(load_raw_transactions())
        report = summarize_splits(chronological_split(cleaned)).set_index("split")
        expected = {
            "train": (0, 125, 396_332, 5_040, 391_292),
            "validation": (126, 152, 98_194, 1_080, 97_114),
            "test": (153, 179, 100_117, 1_080, 99_037),
        }
        for name, values in expected.items():
            actual = report.loc[name]
            self.assertEqual(
                tuple(int(actual[key]) for key in [
                    "simulation_day_min", "simulation_day_max", "rows", "fraud", "legitimate"
                ]),
                values,
            )

    def test_feature_history_excludes_same_day(self):
        source = pd.DataFrame([
            [0, "C1", "4", "M", 1, "M1", 1, "es_food", 10.0, 0],
            [0, "C1", "4", "M", 1, "M1", 1, "es_food", 20.0, 0],
            [1, "C1", "4", "M", 1, "M1", 1, "es_food", 30.0, 1],
            [1, "C1", "4", "M", 1, "M2", 1, "es_travel", 40.0, 0],
        ], columns=RAW_COLUMNS)
        engineered = engineer_features(clean_transactions(source))
        day_zero = engineered.step.eq(0)
        day_one = engineered.step.eq(1)
        self.assertTrue(engineered.loc[day_zero, "customer_previous_transaction_count"].eq(0).all())
        self.assertTrue(engineered.loc[day_one, "customer_previous_transaction_count"].eq(2).all())
        self.assertTrue(engineered.loc[day_one, "customer_previous_avg_spend"].eq(15.0).all())
        self.assertEqual(
            int(engineered.loc[engineered.merchant.eq("M2"), "merchant_previous_transaction_count"].iloc[0]),
            0,
        )

    def test_history_reference_is_label_free_and_count_exact(self):
        cleaned = clean_transactions(load_raw_transactions())
        train = chronological_split(cleaned).train
        reference = build_training_history_reference(train)
        self.assertEqual(set(reference), {
            "customers.parquet", "merchants.parquet", "categories.parquet",
            "customer_merchants.parquet", "customer_categories.parquet",
        })
        self.assertTrue(all("fraud" not in frame for frame in reference.values()))
        self.assertTrue(all(int(frame.transaction_count.sum()) == len(train) for frame in reference.values()))

    def test_active_artifact_paths_and_dashboard_loading(self):
        for path in (
            PROCESSED_TRANSACTIONS_PARQUET,
            TRAIN_SPLIT_PARQUET,
            VALIDATION_SPLIT_PARQUET,
            TEST_SPLIT_PARQUET,
            DASHBOARD_PARQUET,
            PIPELINE_MANIFEST,
            TRAINING_HISTORY_DIR / "metadata.json",
            TRAINING_HISTORY_DIR / "customers.parquet",
            TRAINING_HISTORY_DIR / "merchants.parquet",
            TRAINING_HISTORY_DIR / "categories.parquet",
            TRAINING_HISTORY_DIR / "customer_merchants.parquet",
            TRAINING_HISTORY_DIR / "customer_categories.parquet",
        ):
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.is_relative_to(PROJECT_ROOT))
        dashboard = load_dashboard_data(("step", "fraud"))
        self.assertEqual(len(dashboard), 594_643)
        self.assertEqual((int(dashboard.step.min()), int(dashboard.step.max())), (0, 179))

    def test_frozen_validation_regression_against_original_bundle(self):
        active = load_validation_demo(("source_index", "fraud_probability", "prediction"))
        with zipfile.ZipFile(BUNDLE_ARCHIVE) as archive:
            original_bytes = archive.read("data/validation_demo.parquet")
        original = pd.read_parquet(io.BytesIO(original_bytes), columns=[
            "source_index", "fraud_probability", "prediction"
        ])
        pd.testing.assert_series_equal(active.source_index, original.source_index, check_names=False)
        self.assertLessEqual(
            float(np.max(np.abs(active.fraud_probability.to_numpy() - original.fraud_probability.to_numpy()))),
            1e-12,
        )
        self.assertEqual(int((active.prediction.to_numpy() != original.prediction.to_numpy()).sum()), 0)
        manifest = json.loads(PIPELINE_MANIFEST.read_text())
        self.assertEqual(manifest["split_definition"], SPLIT_CONFIG.rule)
        self.assertEqual(FROZEN_DECISION_THRESHOLD, 0.7935502066058817)

    def test_active_validation_export_has_no_false_time_or_retired_confidence_fields(self):
        columns = set(load_validation_demo().columns)
        self.assertNotIn("day", columns)
        self.assertNotIn("hour_of_day", columns)
        self.assertNotIn("prediction_confidence", columns)
        self.assertNotIn("confidence_level", columns)


if __name__ == "__main__":
    unittest.main()
