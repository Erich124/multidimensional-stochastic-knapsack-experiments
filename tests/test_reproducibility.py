import csv
import tempfile
import unittest
from pathlib import Path

from experiments.generators import explicit_overflow_instance
from experiments.make_tables import (
    _read_float_rows,
    _table_publication_dimension_n6,
    _table_publication_overall,
)
from experiments.study_utils import append_csv, reset_owned_outputs


class ReproducibilityTests(unittest.TestCase):
    def test_runner_owned_csv_reset_prevents_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "owned.csv"
            fields = ["instance", "value"]
            row = {"instance": "a", "value": 1}

            reset_owned_outputs([str(path)])
            append_csv(str(path), row, fields)
            reset_owned_outputs([str(path)])
            append_csv(str(path), row, fields)

            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows, [{"instance": "a", "value": "1"}])

    def test_explicit_overflow_generator_overflows_every_item(self):
        for dims in (2, 3, 4, 5):
            for reward_variant in ("independent", "size_correlated"):
                instance = explicit_overflow_instance(dims, 6, 930000 + dims, reward_variant)
                for item in instance.items:
                    p_over = sum(outcome.prob for outcome in item.outcomes if max(outcome.size) > 1.0)
                    self.assertGreater(p_over, 0.0)

    def test_publication_overall_table_uses_all_exact_rows(self):
        rows = _read_float_rows("results/exact/raw_exact.csv")
        table = _table_publication_overall(rows)
        self.assertEqual(len(rows), 3600)
        self.assertTrue(all(int(row["Observations"]) == 3600 for row in table))

    def test_publication_dimension_table_uses_matched_n6_rows(self):
        rows = _read_float_rows("results/exact/raw_exact.csv")
        table = _table_publication_dimension_n6(rows)
        self.assertEqual({int(row["d"]) for row in table}, {2, 3, 4, 5})
        self.assertTrue(all(int(row["Observations"]) == 300 for row in table))


if __name__ == "__main__":
    unittest.main()
