from __future__ import annotations

import json
import unittest

import pandas as pd

from fraud_pipeline.config import HISTORY_SCHEMA_VERSION, SPLIT_CONFIG
from utils.data_loader import load_validation_transaction
from utils.history_context import build_historical_context, load_training_history
from utils.paths import DASHBOARD_PARQUET, TRAINING_HISTORY_DIR


class HistoricalContextTests(unittest.TestCase):
    def test_reference_lineage_is_training_only(self):
        metadata = json.loads((TRAINING_HISTORY_DIR / "metadata.json").read_text())
        self.assertEqual(metadata["schema_version"], HISTORY_SCHEMA_VERSION)
        self.assertEqual(metadata["split_definition"], SPLIT_CONFIG.rule)
        self.assertEqual(metadata["training_step_min"], 0)
        self.assertEqual(metadata["training_step_max"], 125)
        self.assertEqual(metadata["training_unique_steps"], 126)
        self.assertEqual(metadata["training_row_count"], 396_332)
        self.assertEqual(metadata["excluded_fields"], ["fraud"])

        reference = load_training_history()
        self.assertEqual(int(reference.customers.transaction_count.sum()), 396_332)
        self.assertEqual(int(reference.merchants.transaction_count.sum()), 396_332)
        self.assertEqual(int(reference.categories.transaction_count.sum()), 396_332)
        self.assertEqual(int(reference.customer_merchants.sum()), 396_332)
        self.assertEqual(int(reference.customer_categories.sum()), 396_332)

    def test_known_counts_and_category_denominator_match_training_source(self):
        row = load_validation_transaction(396332).iloc[0]
        context = build_historical_context(row)
        training = pd.read_parquet(
            DASHBOARD_PARQUET,
            columns=["step", "customer", "merchant", "category"],
            filters=[("step", "<=", 125)],
        )
        customer_rows = training.customer.eq(row.customer)
        customer_count = int(customer_rows.sum())
        pair_count = int((customer_rows & training.merchant.eq(row.merchant)).sum())
        category_count = int((customer_rows & training.category.eq(row.category)).sum())

        self.assertEqual(context.entity.customer.transaction_count, customer_count)
        self.assertEqual(context.entity.customer_merchant_count, pair_count)
        self.assertEqual(context.entity.customer_category_count, category_count)
        self.assertAlmostEqual(context.behavior.customer_category_frequency, category_count / customer_count)

    def test_unseen_customer_and_relationships_are_safe_zeros(self):
        row = pd.Series({
            "customer": "__never_seen_customer__",
            "merchant": "__never_seen_merchant__",
            "category": "__never_seen_category__",
            "amount": 100.0,
        })
        context = build_historical_context(row)
        self.assertEqual(context.entity.customer.transaction_count, 0)
        self.assertEqual(context.entity.customer_merchant_count, 0)
        self.assertEqual(context.entity.customer_category_count, 0)
        self.assertIsNone(context.behavior.customer_category_frequency)
        self.assertIsNone(context.behavior.customer_amount_ratio)
        self.assertIsNone(context.behavior.merchant_amount_ratio)
        self.assertIsNone(context.behavior.category_amount_ratio)

    def test_unseen_merchant_and_unseen_pair_are_independent(self):
        reference = load_training_history()
        known_customer = str(reference.customers.index[0])
        known_merchant = str(reference.merchants.index[0])
        known_category = str(reference.categories.index[0])

        unseen_merchant = build_historical_context(pd.Series({
            "customer": known_customer,
            "merchant": "__never_seen_merchant__",
            "category": known_category,
            "amount": 100.0,
        }))
        self.assertGreater(unseen_merchant.entity.customer.transaction_count, 0)
        self.assertEqual(unseen_merchant.entity.merchant.transaction_count, 0)
        self.assertEqual(unseen_merchant.entity.customer_merchant_count, 0)
        self.assertIsNone(unseen_merchant.behavior.merchant_amount_ratio)

        pair_index = reference.customer_merchants.index
        customer_merchants = set(pair_index.get_level_values("merchant")[
            pair_index.get_level_values("customer") == known_customer
        ])
        alternate_merchant = next(
            merchant for merchant in reference.merchants.index
            if merchant not in customer_merchants
        )
        unseen_pair = build_historical_context(pd.Series({
            "customer": known_customer,
            "merchant": alternate_merchant,
            "category": known_category,
            "amount": 100.0,
        }))
        self.assertGreater(unseen_pair.entity.merchant.transaction_count, 0)
        self.assertEqual(unseen_pair.entity.customer_merchant_count, 0)

    def test_amount_history_statistics_match_training_reference(self):
        reference = load_training_history()
        row = load_validation_transaction(396332).iloc[0]
        training_amounts = pd.read_parquet(
            DASHBOARD_PARQUET,
            columns=["step", "customer", "amount"],
            filters=[("step", "<=", 125)],
        )
        amounts = training_amounts.loc[
            training_amounts.customer.eq(row.customer), "amount"
        ]
        context = build_historical_context(row)
        self.assertEqual(context.entity.customer.transaction_count, len(amounts))
        self.assertAlmostEqual(context.entity.customer.amount_mean, float(amounts.mean()))
        self.assertEqual(context.entity.customer.amount_min, float(amounts.min()))
        self.assertEqual(context.entity.customer.amount_max, float(amounts.max()))


if __name__ == "__main__":
    unittest.main()
