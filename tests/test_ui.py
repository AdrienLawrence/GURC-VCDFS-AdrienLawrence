"""Read-only UI data and Pareto calculations; no graphical window is opened."""

import json
import tempfile
import unittest
from pathlib import Path

from schemas import create_artifact
from ui import load_summary, pareto_rows
from ui.pareto import quality_marker_sizes, resource_value


class ParetoViewTests(unittest.TestCase):
    def rows(self):
        return [
            {"algorithm": "fast", "mean_regret_pp": 2.0, "mean_runtime_seconds": 1.0},
            {"algorithm": "balanced", "mean_regret_pp": 1.0, "mean_runtime_seconds": 2.0},
            {"algorithm": "slow-best", "mean_regret_pp": 0.5, "mean_runtime_seconds": 3.0},
            {"algorithm": "dominated", "mean_regret_pp": 2.5, "mean_runtime_seconds": 2.5},
        ]

    def test_extracts_nondominated_frontier(self):
        frontier = pareto_rows(self.rows(), "mean_runtime_seconds")
        self.assertEqual([row["algorithm"] for row in frontier], ["fast", "balanced", "slow-best"])

    def test_better_regret_receives_more_visual_weight(self):
        sizes = quality_marker_sizes([0.1, 1.0, 2.0])
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreater(sizes[1], sizes[2])
        self.assertEqual(quality_marker_sizes([2.0, 2.0]), [100.0, 100.0])

    def test_memory_time_product_uses_mib_seconds(self):
        row = {
            "mean_runtime_seconds": 2.5,
            "mean_rss_peak_delta_bytes": 2 * 1024**2,
        }
        self.assertEqual(resource_value(row, "memory_time_product"), 5.0)
        self.assertIsNone(resource_value({}, "memory_time_product"))

    def test_loading_summary_does_not_modify_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(create_artifact("search_summary", self.rows()).to_dict()),
                encoding="utf-8",
            )
            before = path.read_bytes()
            rows = load_summary(path)
            pareto_rows(rows, "mean_runtime_seconds")
            after = path.read_bytes()
        self.assertEqual(before, after)

    def test_rejects_old_search_summary_schema(self):
        document = create_artifact("search_summary", self.rows()).to_dict()
        document["schema_version"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_summary(path)


if __name__ == "__main__":
    unittest.main()
