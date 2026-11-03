"""Focused checks for the NB301 study protocol; no model training or downloads."""

import queue
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from algs import Budget, dfs
from bench301.behavior.analysis import anytime_mean_error, split_aware_summary
from bench301.behavior.experiment import ResourceGuard, open_or_resume, run_one
from bench301.behavior.presets import parameter_specs
from bench301.heuristic.model import FrozenPredictor, PrefixTransformer
from bench301.oracle import _Bridge
from bench301.representation import random_complete_state
from bench301.space import ArchitectureTree
from schemas import SchemaValidationError, create_artifact, write_artifact
from search_registry import AlgorithmSpec


class CheapOracle:
    def __init__(self):
        self.cache = {}
        self.surrogate_calls = 0

    def clear(self):
        self.cache.clear()
        self.surrogate_calls = 0

    def __call__(self, state):
        if state not in self.cache:
            self.cache[state] = 9.0 + sum(state) / 1000
            self.surrogate_calls += 1
        return self.cache[state]


LIMITS = {"seconds": 30, "rss_delta_mib": 512, "expansions": 100}


class BehaviorTests(unittest.TestCase):
    def test_presets_are_disjoint_and_dispatchable(self):
        family = parameter_specs("v-family")
        baselines = parameter_specs("baselines")
        self.assertEqual(len(family), 20)
        self.assertEqual(len(baselines), 13)
        self.assertEqual(len({row.name for row in family + baselines}), 33)
        tree = ArchitectureTree(lambda state: float(sum(state)), lambda state: 6.0)
        from search_registry import execute

        for spec in family + baselines:
            result = execute(spec, tree, Budget(expansions=2, queries=1))
            self.assertLessEqual(result.expanded, 2, spec.name)

    def test_guarded_no_terminal_is_a_censored_row(self):
        row = run_one(
            AlgorithmSpec("BFS", "bfs", {}),
            -1,
            lambda state: 6.0,
            CheapOracle(),
            {**LIMITS, "expansions": 1},
            8,
        )
        self.assertFalse(row["found"])
        self.assertIsNone(row["best_error"])
        self.assertIsNone(row["anytime_mean_error"])
        self.assertEqual(row["queries"], 0)
        self.assertEqual(row["expanded"], 1)
        self.assertEqual(row["stop"], "expansion_budget")
        self.assertEqual(row["budget_mode"], "guarded_exhaustive")

    def test_wide_breadth_search_stops_before_frontier_explosion(self):
        row = run_one(
            AlgorithmSpec("VCBFS-A-1-2", "vcbfsa", {"kmin": 1, "kmax": 2}),
            -1,
            lambda state: float(sum(state)),
            CheapOracle(),
            {**LIMITS, "expansions": 30},
            8,
        )
        self.assertEqual(row["expanded"], 30)
        self.assertEqual(row["stop"], "expansion_budget")
        self.assertFalse(row["found"])
        self.assertGreater(row["generated"], row["expanded"])

    def test_first_terminal_and_fixed_horizon(self):
        row = run_one(
            AlgorithmSpec("DFS", "dfs", {}),
            1,
            lambda state: 6.0,
            CheapOracle(),
            LIMITS,
            1,
        )
        self.assertTrue(row["found"])
        self.assertEqual(row["queries"], 1)
        self.assertEqual(row["stop"], "query_budget")
        self.assertEqual(row["anytime_mean_error"], row["best_error"])
        self.assertEqual(len(row["best_state"]), 22)
        self.assertEqual(row["expanded"], 22)
        self.assertNotIn("optimum_found", row)

    def test_external_stop_is_returned_without_changing_old_budgets(self):
        tree = ArchitectureTree(lambda state: 6.0, lambda state: 6.0)
        self.assertEqual(dfs(tree, budget=Budget(stop_if=lambda: "time_limit")).stop, "time_limit")
        self.assertEqual(dfs(tree, budget=Budget(expansions=1)).stop, "expansion_budget")

    def test_resource_guard_and_worker_timeout(self):
        with patch("bench301.behavior.experiment.current_rss_bytes", side_effect=[100, 200]):
            self.assertEqual(ResourceGuard(None, 0.00005)(), "memory_limit")
        self.assertEqual(ResourceGuard(0, 512)(), "time_limit")
        bridge = object.__new__(_Bridge)
        bridge._lines = queue.Queue()
        bridge.process = Mock()
        with self.assertRaisesRegex(TimeoutError, "prediction"):
            bridge._receive(0.001, "prediction")
        bridge.process.terminate.assert_called_once()

    def test_optional_batched_scoring_matches_scalar_result(self):
        class Predictor:
            def __init__(self):
                self.batches = 0
                self.scalars = 0

            def __call__(self, state):
                self.scalars += 1
                return float(sum(state))

            def predict(self, states):
                self.batches += 1
                return [float(sum(state)) for state in states]

        predictor = Predictor()
        from search_registry import execute

        spec = AlgorithmSpec("CDFS-2", "cdfs", {"k": 2})
        batched = execute(
            spec,
            ArchitectureTree(predictor, lambda state: float(sum(state)), batch_scores=True),
            Budget(queries=1),
        )
        scalar = execute(
            spec,
            ArchitectureTree(lambda state: float(sum(state)), lambda state: float(sum(state))),
            Budget(queries=1),
        )
        self.assertEqual(batched, scalar)
        self.assertGreater(predictor.batches, 0)
        self.assertEqual(predictor.scalars, 0)

    def test_frozen_transformer_batch_scores_match_scalar_scores(self):
        predictor = FrozenPredictor(PrefixTransformer(), 5.0, 2.0)
        rng = random.Random(71)
        prefixes = [random_complete_state(rng)[: rng.randrange(1, 23)] for _ in range(50)]
        batched = predictor.predict(prefixes)
        scalar = [predictor(state) for state in prefixes]
        self.assertTrue(
            all(
                abs(float(batch) - single) < 1e-5
                for batch, single in zip(batched, scalar, strict=True)
            )
        )

    def test_anytime_carry_forward_and_split_aggregation(self):
        trace = [{"query": 1, "value": 8.0}, {"query": 3, "value": 6.0}]
        self.assertEqual(anytime_mean_error(trace, 4), 7.0)
        sample = run_one(
            AlgorithmSpec("DFS", "dfs", {}),
            1,
            lambda state: 6.0,
            CheapOracle(),
            LIMITS,
            1,
        )
        sample.update(checkpoint="split-101-model-0", split_seed=101, model_seed=0)
        other = dict(sample, checkpoint="split-203-model-0", split_seed=203)
        per_split, summary = split_aware_summary([sample, other])
        self.assertEqual(len(per_split), 2)
        self.assertEqual(summary[0]["independent_splits"], 2)
        self.assertEqual(summary[0]["valid_runs"], 2)

    def test_resume_preserves_existing_rows_and_rejects_plan_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new-study"
            manifest = {"test_plan": "fixed"}
            planned = [("split-101-model-0", "DFS", 1)]
            self.assertEqual(open_or_resume(path, manifest, planned), [])
            row = run_one(
                AlgorithmSpec("DFS", "dfs", {}),
                1,
                lambda state: 6.0,
                CheapOracle(),
                LIMITS,
                1,
            )
            row.update(checkpoint="split-101-model-0", split_seed=101, model_seed=0)
            rows = [row]
            write_artifact(path / "runs.partial.json", "nb301_behavior_runs", rows)
            self.assertEqual(open_or_resume(path, manifest, planned), rows)
            with self.assertRaisesRegex(ValueError, "changed study plan"):
                open_or_resume(path, {"test_plan": "changed"}, planned)

    def test_behavior_artifact_rejects_malformed_or_inconsistent_rows(self):
        with self.assertRaises(SchemaValidationError):
            create_artifact("nb301_behavior_runs", [1])
        row = run_one(
            AlgorithmSpec("DFS", "dfs", {}),
            1,
            lambda state: 6.0,
            CheapOracle(),
            LIMITS,
            1,
        )
        row.update(checkpoint="split-101-model-0", split_seed=101, model_seed=0)
        row["found"] = False
        with self.assertRaises(SchemaValidationError):
            create_artifact("nb301_behavior_runs", [row])


if __name__ == "__main__":
    unittest.main()
