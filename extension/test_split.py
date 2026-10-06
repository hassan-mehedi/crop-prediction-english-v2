"""Verify holdout isolation, repeatability, and stale-split protection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from extension.split import DATASET_PATH, load_training_data, prepare_split


class EvaluationSplitTests(unittest.TestCase):
    def test_holdout_and_cv_are_disjoint_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            data, saved = prepare_split(split_path=path)
            train, test = set(saved["train_rows"]), set(saved["final_test_rows"])
            self.assertFalse(train & test)
            self.assertEqual(train | test, set(range(len(data))))
            self.assertEqual(len(train), 1760)
            self.assertEqual(len(test), 440)
            self.assertEqual(set(saved["training_class_counts"].values()), {80})
            self.assertEqual(set(saved["final_test_class_counts"].values()), {20})
            X, y, cv = load_training_data(split_path=path)
            self.assertEqual(len(X), 1760)
            all_validation = []
            for fit, validation in cv:
                self.assertFalse(set(fit) & set(validation))
                self.assertEqual(set(fit) | set(validation), set(range(len(X))))
                self.assertEqual(set(y.iloc[validation].value_counts()), {16})
                original_rows = np.asarray(saved["train_rows"])[np.r_[fit, validation]]
                self.assertFalse(set(original_rows) & test)
                all_validation.extend(validation.tolist())
            self.assertEqual(sorted(all_validation), list(range(len(X))))

    def test_repeat_run_preserves_saved_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            _, first = prepare_split(split_path=path)
            original = path.read_bytes()
            _, second = prepare_split(split_path=path)
            self.assertEqual(first, second)
            self.assertEqual(original, path.read_bytes())

    def test_changed_dataset_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.csv"
            path = Path(directory) / "split.json"
            dataset.write_bytes(DATASET_PATH.read_bytes())
            prepare_split(dataset, path)
            original = path.read_bytes()
            dataset.write_bytes(dataset.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Saved split differs"):
                prepare_split(dataset, path)
            self.assertEqual(original, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
