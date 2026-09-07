from __future__ import annotations

import unittest

import numpy as np

from utils.artifact_loader import ModelArtifacts, load_fraud_model, load_model_config
from utils.data_loader import load_validation_index, load_validation_transaction
from utils.inference import assess_transaction, predict_transactions


class DecisionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = load_validation_index()
        ids = [int(index.iloc[0].source_index)]
        ids.append(int(index.loc[index.prediction.eq(1), "source_index"].iloc[0]))
        cls.rows = [load_validation_transaction(source_id) for source_id in ids]

    def test_frozen_probability_prediction_and_threshold_are_unchanged(self):
        expected_threshold = float(load_model_config()["decision_threshold"])
        for row in self.rows:
            with self.subTest(source_index=int(row.source_index.iloc[0])):
                output = predict_transactions(row).iloc[0]
                self.assertAlmostEqual(output.fraud_probability, float(row.fraud_probability.iloc[0]), places=12)
                self.assertEqual(int(output.prediction), int(row.prediction.iloc[0]))
                self.assertEqual(output.decision_threshold, expected_threshold)

    def test_tree_and_leaf_evidence_has_literal_bounds(self):
        tree_count = len(load_fraud_model().estimators_)
        for row in self.rows:
            evidence = assess_transaction(row).model
            self.assertGreaterEqual(evidence.tree_agreement, 0)
            self.assertLessEqual(evidence.tree_agreement, 1)
            self.assertLessEqual(evidence.tree_agreement_count, tree_count)
            self.assertEqual(evidence.tree_count, tree_count)
            self.assertGreaterEqual(evidence.leaf_training_support.minimum, 0)
            self.assertGreaterEqual(evidence.leaf_training_support.maximum, evidence.leaf_training_support.minimum)

    def test_decision_margin_sign_matches_decision(self):
        for row in self.rows:
            evidence = assess_transaction(row).model
            self.assertEqual(evidence.decision_margin >= 0, bool(evidence.prediction))
            self.assertTrue(np.isclose(
                evidence.decision_margin,
                evidence.fraud_probability - evidence.decision_threshold,
            ))

    def test_secondary_confidence_model_is_not_a_runtime_artifact(self):
        self.assertNotIn("confidence_model", ModelArtifacts.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
