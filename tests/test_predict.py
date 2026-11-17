import numpy as np
import pandas as pd
import unittest

from src import predict


def make_batch():
    row = {column: 0.0 for column in predict.REQUIRED_INPUT_COLUMNS}
    row["Amount"] = 12.5
    row["transaction_id"] = "txn-1"
    return pd.DataFrame([row])


class PredictBatchTests(unittest.TestCase):
    def test_validate_batch_input_checks_required_features_and_labels(self):
        with self.assertRaisesRegex(ValueError, "Time"):
            predict.validate_batch_input(make_batch().drop(columns="Time"))

        batch = make_batch()
        batch["Class"] = [2]
        with self.assertRaisesRegex(ValueError, "Class"):
            predict.validate_batch_input(batch)

    def test_predict_batch_excludes_labels_and_preserves_transaction_columns(self):
        expected_features = predict.REQUIRED_INPUT_COLUMNS + [
            "TransactionHour",
            "TransactionDay",
        ]

        class FixedModel:
            feature_names_in_ = np.array(expected_features)

            def predict_proba(self, values):
                if list(values.columns) != expected_features:
                    raise AssertionError("The model received unexpected feature columns.")
                return np.tile([0.2, 0.8], (len(values), 1))

        batch = make_batch()
        batch["Class"] = [1]
        result = predict.predict_batch(batch, FixedModel(), threshold=0.5)

        self.assertEqual(result.loc[0, "transaction_id"], "txn-1")
        self.assertAlmostEqual(result.loc[0, "proba_fraude"], 0.8)
        self.assertEqual(result.loc[0, "classe_predite"], 1)
        self.assertEqual(result.loc[0, "Class"], 1)

    def test_validate_batch_input_rejects_invalid_numeric_values_and_amounts(self):
        batch = make_batch()
        batch["V1"] = ["not-a-number"]
        with self.assertRaisesRegex(ValueError, "V1"):
            predict.validate_batch_input(batch)

        batch = make_batch()
        batch.loc[0, "Amount"] = -1
        with self.assertRaisesRegex(ValueError, "montants"):
            predict.validate_batch_input(batch)


if __name__ == "__main__":
    unittest.main()
