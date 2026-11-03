"""Empirical runner tests use synthetic trees and records, never NAS claims."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from algs import Budget
from bench201.empirical import (
    measure_search,
)
from bench201.specs import DEFAULT_SPECS, AlgorithmSpec, execute, load_specs
from run import DemoTree
from search_reporting import (
    json_histogram,
    print_final_summary,
    print_run,
    quality_metrics,
    summarize,
)


class EmpiricalRunnerTests(unittest.TestCase):
    def test_run_rows_use_json_safe_width_histogram_keys(self):
        histogram = json_histogram({4: 1, 2: 3})
        self.assertEqual(histogram, {"2": 3, "4": 1})
        self.assertTrue(all(isinstance(key, str) for key in histogram))

    def test_primary_regret_is_search_only(self):
        quality = quality_metrics(search_best=9.0, initial_incumbent=8.0, oracle_minimum=7.0)
        self.assertEqual(quality["reported_best_error"], 9.0)
        self.assertEqual(quality["regret_pp"], 2.0)
        self.assertEqual(quality["incumbent_augmented_best_error"], 8.0)
        self.assertEqual(quality["incumbent_augmented_regret_pp"], 1.0)
        with self.assertRaises(RuntimeError):
            quality_metrics(None, 8.0, 7.0)

    def test_default_specs_include_both_dynamic_beam_controllers(self):
        dynamic = [spec for spec in DEFAULT_SPECS if spec.algorithm == "dynamic_beam"]
        self.assertEqual(
            {spec.parameters["policy"] for spec in dynamic},
            {"entropy", "standard_deviation"},
        )

    def test_corrected_and_historical_vcbfs_are_distinct_defaults(self):
        names = {(spec.name, spec.algorithm) for spec in DEFAULT_SPECS}
        self.assertIn(("VCBFS-20-40pct", "vcbfs"), names)
        self.assertIn(("VCBFSBeam-20-40pct", "vcbfs_beam"), names)

    def test_every_default_algorithm_spec_executes(self):
        for spec in DEFAULT_SPECS:
            with self.subTest(spec=spec.name):
                result = execute(spec, DemoTree(), Budget(queries=2))
                self.assertLessEqual(result.queries, 2)

    def test_measurement_replay_and_counters(self):
        spec = AlgorithmSpec("Beam-2", "beam", {"width": 2})
        result, events, elapsed, peak, rss_peak, rss_delta = measure_search(
            spec, DemoTree, budget=Budget()
        )
        self.assertEqual(result.queries, 2)
        self.assertEqual(len(events.terminals), 2)
        self.assertGreaterEqual(elapsed, 0)
        self.assertGreater(peak, 0)
        self.assertTrue(rss_peak is None or rss_peak > 0)
        self.assertTrue(rss_delta is None or rss_delta >= 0)

    def test_custom_specs_validate_schema_and_dynamic_policy(self):
        payload = [
            {
                "name": "DBS-E",
                "algorithm": "dynamic_beam",
                "parameters": {
                    "policy": "entropy",
                    "minimum_width": 1,
                    "maximum_width": 4,
                    "slope": 1,
                    "intercept": 1,
                    "temperature": 1,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specs.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_specs(path)[0].algorithm, "dynamic_beam")
            path.write_text(json.dumps([{"name": "bad"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_specs(path)

    def test_summary_uses_split_means(self):
        rows = [
            {"algorithm": "A", "split_seed": split, **{key: value for key in METRICS}}
            for split, value in ((1, 1.0), (1, 3.0), (2, 5.0))
        ]
        per_split, final = summarize(rows)
        self.assertEqual(len(per_split), 2)
        self.assertEqual(final[0]["independent_splits"], 2)
        self.assertEqual(final[0]["mean_regret_pp"], 3.5)

    def test_console_reporting_is_human_readable(self):
        row = {
            "algorithm": "A",
            "split_seed": 1,
            "model_seed": 2,
            "reported_best_error": 9.0,
            "regret_pp": 0.5,
            "queries": 5,
            "query_budget": 5,
            "expanded": 10,
            "scored": 20,
            "pruned": 4,
            "peak_pending_entries": 3,
            "runtime_seconds": 0.1,
            "python_peak_bytes": 1024,
            "rss_peak_delta_bytes": 2048,
            "stop": "query_budget",
        }
        summary = {
            "algorithm": "A",
            "independent_splits": 1,
            "mean_regret_pp": 0.5,
            "mean_runtime_seconds": 0.1,
            "mean_expanded": 10,
            "mean_scored": 20,
            "mean_queries": 5,
            "mean_pruned": 4,
            "mean_peak_pending_entries": 3,
            "mean_rss_peak_delta_bytes": 2048,
        }
        output = io.StringIO()
        with redirect_stdout(output):
            print_run(row)
            print_final_summary([summary])
        rendered = output.getvalue()
        self.assertIn("RESULT A", rendered)
        self.assertIn("FINAL SPLIT-AWARE SUMMARY", rendered)

    def test_final_summary_ranks_regret_then_memory_then_runtime(self):
        def row(name, regret, memory, runtime):
            return {
                "algorithm": name,
                "independent_splits": 1,
                "mean_regret_pp": regret,
                "mean_runtime_seconds": runtime,
                "mean_expanded": 10,
                "mean_scored": 20,
                "mean_queries": 5,
                "mean_pruned": 4,
                "mean_peak_pending_entries": 3,
                "mean_rss_peak_delta_bytes": memory,
            }

        output = io.StringIO()
        with redirect_stdout(output):
            print_final_summary(
                [
                    row("fast", 0, 2048, 0.01),
                    row("memory", 0, 1024, 10),
                    row("quality", 0.1, 0, 0),
                ]
            )
        rendered = output.getvalue()
        self.assertLess(rendered.index("memory"), rendered.index("fast"))
        self.assertLess(rendered.index("fast"), rendered.index("quality"))


METRICS = (
    "regret_pp",
    "runtime_seconds",
    "python_peak_bytes",
    "rss_peak_bytes",
    "rss_peak_delta_bytes",
    "peak_pending_entries",
    "expanded",
    "generated",
    "scored",
    "queries",
    "pruned",
)


if __name__ == "__main__":
    unittest.main()
