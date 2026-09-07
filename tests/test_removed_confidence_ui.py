from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RemovedConfidenceUiTests(unittest.TestCase):
    def test_active_ui_has_no_correctness_probability_or_p95_support(self):
        active_files = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py")), *sorted((ROOT / "components").glob("*.py"))]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in active_files).lower()
        forbidden = [
            "prediction_confidence",
            "probability the prediction is correct",
            "very high confidence",
            "95th-percentile reference",
            "confidence_model",
        ]
        for phrase in forbidden:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, combined)


if __name__ == "__main__":
    unittest.main()
