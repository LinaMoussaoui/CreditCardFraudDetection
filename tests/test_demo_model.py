import unittest

import pandas as pd

from src import demo_model


class DemoModelTests(unittest.TestCase):
    def test_synthetic_training_data_is_reproducible_and_readable(self):
        first = demo_model.make_synthetic_transactions(n_samples=100)
        second = demo_model.make_synthetic_transactions(n_samples=100)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(set(demo_model.INPUT_COLUMNS).issubset(first.columns))
        self.assertTrue(first["fraude"].isin([0, 1]).all())

    def test_scoring_returns_probabilities_and_alerts(self):
        model = demo_model.train_demo_model()
        examples = pd.DataFrame(
            [
                [35.0, "paiement_carte", 14, False, False],
                [2400.0, "virement", 2, True, True],
            ],
            columns=demo_model.INPUT_COLUMNS,
        )

        results = demo_model.score_transactions(model, examples, threshold=0.5)

        self.assertEqual(len(results), 2)
        self.assertTrue(results["probabilite_fraude"].between(0, 1).all())
        self.assertEqual(results.loc[1, "alerte"], True)
        self.assertIn("decision", results.columns)

    def test_invalid_input_is_rejected_with_a_clear_message(self):
        invalid = pd.DataFrame(
            [[12.0, "unknown", 24, "maybe", False]],
            columns=demo_model.INPUT_COLUMNS,
        )
        with self.assertRaisesRegex(ValueError, "heure"):
            demo_model.validate_transactions(invalid)

        invalid["heure"] = 12
        with self.assertRaisesRegex(ValueError, "Unknown transaction type"):
            demo_model.validate_transactions(invalid)


if __name__ == "__main__":
    unittest.main()
