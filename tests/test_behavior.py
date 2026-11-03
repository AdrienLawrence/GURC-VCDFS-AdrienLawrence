"""Behavioral pruning and anytime analysis tests."""

import json
import tempfile
import unittest
from pathlib import Path

from algs import Budget, vcdfsa
from bench201.behavior.analysis import BehaviorObserver, anytime_mean_regret
from bench201.behavior.experiment import open_or_resume, parameter_specs, save_csv
from bench201.specs import DEFAULT_SPECS
from schemas import create_artifact, write_artifact
from search_registry import execute
from ui.behavior import belongs_to_family, load_behavior_summary, nondominated


class TinyTree:
    initial = ()

    def feasible(self, state):
        return True

    def terminal(self, state):
        return len(state) == 2

    def successors(self, state):
        return [state + (value,) for value in range(3)]

    def heuristic(self, state):
        return float(2 - state[-1])

    def objective(self, state):
        return 0.0 if state == (0, 0) else float(1 + sum(state))


class BehaviorTests(unittest.TestCase):
    def test_bench201_baselines_include_every_cdfs_width_from_one_to_five(self):
        expected = {f"CDFS-{width}" for width in range(1, 6)}
        self.assertEqual(
            {spec.name for spec in DEFAULT_SPECS if spec.algorithm == "cdfs"}, expected
        )
        self.assertEqual(
            {spec.name for spec in parameter_specs("baselines") if spec.algorithm == "cdfs"},
            expected,
        )

    def test_csv_export_accepts_fields_added_by_extended_runs(self):
        rows = [{"algorithm": "old"}, {"algorithm": "new", "budget_mode": "bounded"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            save_csv(path, rows)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "algorithm,budget_mode")
            self.assertEqual(lines[1], "old,")
            self.assertEqual(lines[2], "new,bounded")

    def test_behavior_study_resumes_only_an_identical_plan(self):
        manifest = {
            "data_sha256": "a" * 64,
            "preset": "baselines",
            "models_per_split": 1,
            "budgets": [1],
            "checkpoint_directories": ["split-101-model-0"],
            "specifications": [{"name": "DFS", "algorithm": "dfs", "parameters": {}}],
            "metrics_version": 2,
        }
        key = ("split-101-model-0", "DFS", 1)
        row = {"checkpoint": key[0], "algorithm": key[1], "query_budget": key[2]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            self.assertEqual(open_or_resume(output, manifest, [key]), [])
            write_artifact(output / "runs.partial.json", "behavior_runs", [row])
            self.assertEqual(open_or_resume(output, manifest, [key]), [row])
            changed = {**manifest, "budgets": [2]}
            with self.assertRaisesRegex(ValueError, "changed fields: budgets"):
                open_or_resume(output, changed, [(key[0], key[1], 2)])

    def test_behavior_extension_preserves_old_rows_and_resumes_new_rows(self):
        base = {
            "data_sha256": "a" * 64,
            "preset": "absolute-ablation",
            "models_per_split": 1,
            "budgets": [1],
            "checkpoint_directories": ["split-101-model-0"],
            "specifications": [{"name": "old", "algorithm": "vcbfsa", "parameters": {}}],
            "metrics_version": 2,
        }
        extension = {
            **base,
            "preset": "vcbfsn-absolute",
            "specifications": [{"name": "new", "algorithm": "vcbfsna", "parameters": {}}],
        }
        old_key = ("split-101-model-0", "old", 1)
        new_key = ("split-101-model-0", "new", 1)
        old_row = {"checkpoint": old_key[0], "algorithm": old_key[1], "query_budget": 1}
        new_row = {"checkpoint": new_key[0], "algorithm": new_key[1], "query_budget": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            open_or_resume(output, base, [old_key])
            write_artifact(output / "runs.json", "behavior_runs", [old_row])
            self.assertEqual(open_or_resume(output, extension, [new_key], extend=True), [old_row])
            write_artifact(output / "runs.partial.json", "behavior_runs", [old_row, new_row])
            self.assertEqual(
                open_or_resume(output, extension, [new_key], extend=True), [old_row, new_row]
            )

    def test_baseline_exhaustive_extension_adds_missing_widths_only(self):
        original_specs = [
            {"name": f"CDFS-{width}", "algorithm": "cdfs", "parameters": {"k": width}}
            for width in (2, 4)
        ]
        expanded_specs = [
            {"name": f"CDFS-{width}", "algorithm": "cdfs", "parameters": {"k": width}}
            for width in range(1, 6)
        ]
        original = {
            "data_sha256": "a" * 64,
            "preset": "baselines",
            "models_per_split": 1,
            "budgets": ["exhaustive"],
            "checkpoint_directories": ["split-101-model-0"],
            "specifications": original_specs,
            "metrics_version": 2,
        }
        expanded = {**original, "specifications": expanded_specs}
        old_keys = [("split-101-model-0", spec["name"], -1) for spec in original_specs]
        all_keys = [("split-101-model-0", spec["name"], -1) for spec in expanded_specs]
        old_rows = [
            {"checkpoint": checkpoint, "algorithm": algorithm, "query_budget": budget}
            for checkpoint, algorithm, budget in old_keys
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            open_or_resume(output, original, old_keys)
            write_artifact(output / "runs.json", "behavior_runs", old_rows)
            self.assertEqual(open_or_resume(output, expanded, all_keys, extend=True), old_rows)
            self.assertEqual(open_or_resume(output, expanded, all_keys, extend=True), old_rows)

    def test_absolute_ablation_contains_every_ordered_bound_pair_for_a_families(self):
        specs = parameter_specs("absolute-ablation")
        expected_pairs = {(lower, upper) for lower in range(1, 6) for upper in range(lower, 6)}
        self.assertEqual(len(specs), 45)
        self.assertEqual({spec.algorithm for spec in specs}, {"vcdfsa", "vcbfsa", "vcbfsna"})
        for algorithm in ("vcdfsa", "vcbfsa", "vcbfsna"):
            self.assertEqual(
                {
                    (spec.parameters["kmin"], spec.parameters["kmax"])
                    for spec in specs
                    if spec.algorithm == algorithm
                },
                expected_pairs,
            )

    def test_every_absolute_ablation_implementation_reaches_dispatcher(self):
        specs = parameter_specs("absolute-ablation")
        representatives = {spec.algorithm: spec for spec in specs}
        for spec in representatives.values():
            result = execute(spec, TinyTree(), Budget(queries=1))
            self.assertEqual(result.queries, 1, spec.algorithm)

    def test_unbounded_budget_exhausts_reachable_tree(self):
        spec = next(spec for spec in parameter_specs("baselines") if spec.algorithm == "dfs")
        result = execute(spec, TinyTree(), Budget())
        self.assertEqual(result.stop, "exhausted")
        self.assertEqual(result.queries, 9)

    def test_baseline_preset_excludes_v_family_and_matches_standard_baselines(self):
        specs = parameter_specs("baselines")
        self.assertEqual(len(specs), 15)
        self.assertEqual(
            {spec.algorithm for spec in specs},
            {
                "dfs",
                "bfs",
                "heuristic_dfs",
                "gbfs",
                "astar",
                "mcts",
                "cdfs",
                "beam",
                "dynamic_beam",
            },
        )
        self.assertFalse(any(spec.algorithm.startswith("vc") for spec in specs))
        self.assertTrue(belongs_to_family({"implementation": "dynamic_beam"}, "Baselines"))
        self.assertFalse(belongs_to_family({"implementation": "vcdfsa"}, "Baselines"))

    def test_behavior_summary_loader_accepts_aggregated_contract(self):
        row = {
            "algorithm": "VCDFS-20-40pct",
            "implementation": "vcdfs",
            "query_budget": 8,
            "mean_final_regret_pp": 1.0,
            "mean_anytime_mean_regret_pp": 1.5,
            "mean_all_optima_pruned": 0.5,
            "mean_optimum_found": 0.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(create_artifact("behavior_summary", [row]).to_dict()),
                encoding="utf-8",
            )
            self.assertEqual(load_behavior_summary(path), [row])

    def test_behavior_frontier_rejects_dominated_configuration(self):
        rows = [
            {"algorithm": "cheap", "work": 1.0, "regret": 2.0},
            {"algorithm": "quality", "work": 2.0, "regret": 1.0},
            {"algorithm": "dominated", "work": 3.0, "regret": 3.0},
        ]
        self.assertEqual(
            [row["algorithm"] for row in nondominated(rows, "work", "regret")],
            ["cheap", "quality"],
        )

    def test_anytime_regret_holds_final_incumbent_to_horizon(self):
        improvements = [{"query": 1, "value": 4.0}, {"query": 3, "value": 2.0}]
        self.assertAlmostEqual(anytime_mean_regret(improvements, 1.0, 4), 2.0)

    def test_observer_detects_pruned_optimal_path_and_best_milestone(self):
        observer = BehaviorObserver({(0, 0)}, oracle=0.0, terminal_depth=2)
        result = vcdfsa(TinyTree(), kmin=1, kmax=1, budget=Budget(queries=1), event=observer)
        metrics = observer.metrics(1)
        self.assertEqual(result.queries, 1)
        self.assertEqual(metrics["any_optimum_pruned"], 1)
        self.assertEqual(metrics["all_optima_pruned"], 1)
        self.assertEqual(metrics["optimal_paths_surviving"], 0)
        self.assertEqual(metrics["query_of_final_best"], 1)
        self.assertEqual(metrics["optimum_found"], 0)
        self.assertEqual(metrics["optimal_prune_events"][0]["depth"], 1)

    def test_observer_tracks_global_beam_scoring_and_optimal_pruning(self):
        observer = BehaviorObserver({(0, 0)}, oracle=0.0, terminal_depth=2)
        observer({"kind": "rank", "count": 3})
        observer(
            {
                "kind": "beam",
                "retained": [{"state": (1,), "score": 0.0}],
                "pruned": [{"state": (0,), "score": 1.0}],
            }
        )
        self.assertEqual(observer.scored, 3)
        self.assertEqual(observer.surviving, set())
        self.assertEqual(observer.optimal_prune_events[0]["depth"], 1)


if __name__ == "__main__":
    unittest.main()
