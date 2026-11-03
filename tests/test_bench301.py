"""Offline NB301 tests: real-format fixtures and deterministic synthetic outcomes."""

import io
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from algs import Budget, Retention, cdfs, dfs, vcdfsa
from bench201.heuristic.model import FrozenPredictor as Predictor201
from bench301.empirical import run_one
from bench301.heuristic.data import PrefixDataset, load_measurements, split_indices, tokenize
from bench301.heuristic.evaluate import evaluate, surrogate_alignment
from bench301.heuristic.extract import extract, parse_record
from bench301.heuristic.model import FrozenPredictor, PrefixTransformer
from bench301.heuristic.train import TrainingConfig, fit
from bench301.oracle import TerminalOracle
from bench301.oracle_bridge import main as bridge_main
from bench301.reference import make_reference
from bench301.representation import (
    BASE,
    BRANCHING_FACTORS,
    OPS,
    PAIR_TO_TOKEN,
    SEQUENCE_LENGTH,
    UNASSIGNED,
    Genotype,
    allowed_tokens,
    branching_factor,
    config_to_state,
    genotype_to_state,
    random_complete_state,
    state_to_config,
    state_to_genotype,
    validate_prefix,
)
from bench301.space import ArchitectureTree
from bench301.specs import DEFAULT_SPECS
from search_registry import execute


def records(size=10):
    rng = random.Random(41)
    return [
        {"index": index, "state": list(random_complete_state(rng)), "error": float(index + 1)}
        for index in range(size)
    ]


def raw_fixture(state, accuracy=93.55):
    """Shape/field conventions verified against the official 100-epoch release."""
    config = state_to_config(state)
    config.update(
        {
            BASE + "init_channels": 32,
            BASE + "layers": 8,
            "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 100,
        }
    )
    return {
        "optimized_hyperparamater_config": config,
        "budget": 100,
        "info": [
            {
                "val_accuracy": accuracy,
                "val_accuracy_final": 93.4,
                "epochs": 96,
                "train_datapoints": 40000,
                "val_datapoints": 10000,
                "dataset_path": "datasets/CIFAR10.csv",
            }
        ],
        "test_accuracy": 99.9,
        "created_by_optimizer": "de",
    }


class RepresentationTests(unittest.TestCase):
    def test_exact_branching_and_position_vocabulary(self):
        self.assertEqual(SEQUENCE_LENGTH, 22)
        self.assertEqual(BRANCHING_FACTORS, (7, 7, 3, 7, 7, 6, 7, 7, 10, 7, 7) * 2)
        self.assertEqual(branching_factor(22), 0)
        self.assertEqual(len(OPS), 7)
        self.assertEqual(len(PAIR_TO_TOKEN), 10)
        for depth in range(22):
            self.assertEqual(branching_factor(depth), len(allowed_tokens(depth)))

    def test_two_hundred_state_genotype_config_round_trips(self):
        rng = random.Random(7)
        for _ in range(200):
            state = random_complete_state(rng)
            genotype = state_to_genotype(state)
            self.assertEqual(genotype_to_state(genotype), state)
            self.assertEqual(config_to_state(state_to_config(state)), state)
            self.assertEqual(genotype.normal_concat, [2, 3, 4, 5])
            self.assertEqual(genotype.reduce_concat, [2, 3, 4, 5])

    def test_pair_canonicalization_preserves_source_operation_association(self):
        state = random_complete_state(random.Random(11))
        genotype = state_to_genotype(state)
        normal = [
            edge
            for offset in range(0, 8, 2)
            for edge in reversed(genotype.normal[offset : offset + 2])
        ]
        reduce = [
            edge
            for offset in range(0, 8, 2)
            for edge in reversed(genotype.reduce[offset : offset + 2])
        ]
        reversed_edges = Genotype(normal, [2, 3, 4, 5], reduce, [2, 3, 4, 5])
        self.assertEqual(genotype_to_state(reversed_edges), state)
        self.assertNotEqual(state[:11], state[11:])

    def test_rejects_wrong_position_prefix_and_duplicate_sources(self):
        with self.assertRaises(ValueError):
            validate_prefix((7,))
        with self.assertRaises(ValueError):
            validate_prefix((0, 0, PAIR_TO_TOKEN[(3, 4)]))
        with self.assertRaises(ValueError):
            state_to_genotype((0,))
        genotype = state_to_genotype(random_complete_state(random.Random(8)))
        genotype.normal[1] = (genotype.normal[1][0], 0)
        with self.assertRaises(ValueError):
            genotype_to_state(genotype)


class TreeTests(unittest.TestCase):
    def test_depth_branching_and_objective_boundary(self):
        called = []
        tree = ArchitectureTree(
            lambda state: float(len(state)), lambda state: called.append(state) or 4.0
        )
        self.assertEqual(len(list(tree.successors(()))), 7)
        state = ()
        for depth in range(22):
            self.assertFalse(tree.terminal(state))
            children = list(tree.successors(state))
            self.assertEqual(len(children), BRANCHING_FACTORS[depth])
            state = children[0]
        self.assertTrue(tree.terminal(state))
        self.assertEqual(list(tree.successors(state)), [])
        with self.assertRaises(ValueError):
            tree.objective(state[:-1])
        self.assertEqual(called, [])
        self.assertEqual(tree.objective(state), 4.0)
        self.assertEqual(called, [state])

    def test_same_algorithms_reach_complete_darts_states(self):
        tree = ArchitectureTree(
            lambda state: float(state[-1]) if state else 0.0, lambda state: float(sum(state))
        )
        for algorithm in (dfs, cdfs, vcdfsa):
            result = algorithm(tree, budget=Budget(queries=2))
            self.assertEqual(result.queries, 2)
            self.assertEqual(len(result.best_state), 22)
            self.assertEqual(result.stop, "query_budget")

    def test_existing_absolute_rule_is_preserved_even_when_bounds_exceed_b(self):
        # User explicitly prohibited the requested phase-3 behavior change.
        scores = [0.0, 0.2, 1.0]
        mean = sum(scores) / 3
        mad = sum(abs(value - mean) for value in scores) / 3
        confidence = min(1.0, (scores[1] - scores[0]) / (mad + 1e-5))
        expected = min(3, max(1, int(confidence * 2 + (1 - confidence) * 10)))
        self.assertEqual(Retention("absolute", 2, 10).width(scores)[0], expected)
        self.assertEqual(Retention("absolute", 2, 10).width([5])[0], 1)


class ExtractionTests(unittest.TestCase):
    def test_selected_validation_target_ignores_test_and_final_labels(self):
        state = random_complete_state(random.Random(10))
        raw = raw_fixture(state)
        parsed, trial = parse_record(raw, "results_1.json")
        self.assertEqual(parsed, state)
        self.assertEqual(trial["validation_accuracy"], 93.55)
        self.assertEqual(trial["validation_accuracy_final"], 93.4)
        self.assertEqual(trial["reported_epoch"], 96)
        self.assertIsNone(trial["seed"])
        self.assertNotIn("test_accuracy", trial)
        raw["test_accuracy"] = -999
        self.assertEqual(parse_record(raw, "results_1.json")[1], trial)

    def test_duplicate_grouping_provenance_and_load_round_trip(self):
        rows = records(3)
        documents = [raw_fixture(tuple(row["state"])) for row in rows]
        documents.append(raw_fixture(tuple(rows[0]["state"]), accuracy=91.55))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "records.zip", Path(directory) / "measurements.json"
            with zipfile.ZipFile(source, "w") as archive:
                for index, document in enumerate(documents):
                    archive.writestr(f"rs/results_{index}.json", json.dumps(document))
            payload = extract(source, output)
            loaded, checksum = load_measurements(output)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(payload["observation_count"], 4)
            self.assertEqual(payload["duplicate_observations"], 1)
            duplicate = next(row for row in loaded if len(row["trials"]) == 2)
            self.assertAlmostEqual(duplicate["error"], 7.45)
            self.assertEqual(len(checksum), 64)
            with self.assertRaises(FileExistsError):
                extract(source, output)

    def test_malformed_records_fail_closed(self):
        state = random_complete_state(random.Random(12))
        for case in ("metric", "architecture", "fidelity", "dataset"):
            raw = raw_fixture(state)
            if case == "metric":
                raw["info"][0]["val_accuracy"] = float("nan")
            elif case == "architecture":
                raw["optimized_hyperparamater_config"][BASE + "edge_normal_0"] = "none"
            elif case == "fidelity":
                raw["budget"] = 50
            else:
                raw["info"][0]["dataset_path"] = "other.csv"
            with self.assertRaises(ValueError):
                parse_record(raw, "bad.json")


class HeuristicTests(unittest.TestCase):
    def test_splits_are_disjoint_reproducible_and_configurable(self):
        first = split_indices(100, 21)
        self.assertEqual([len(first[key]) for key in ("train", "dev", "test")], [80, 10, 10])
        self.assertEqual(first, split_indices(100, 21))
        self.assertNotEqual(first, split_indices(100, 22))
        self.assertEqual(len(set(first["train"] + first["dev"] + first["test"])), 100)
        self.assertEqual(len(split_indices(100, 21, 12, 8)["train"]), 12)

    def test_lazy_prefix_dataset_uses_22_equal_weight_depths(self):
        rows = records(3)
        dataset = PrefixDataset(rows)
        self.assertEqual(len(dataset), 66)
        self.assertEqual(dataset.states.shape, (3, 22))
        for index in range(22):
            tokens, target = dataset[index]
            self.assertEqual(tokens.tolist(), list(tokenize(tuple(rows[0]["state"][: index + 1]))))
            self.assertEqual(float(target), rows[0]["error"])
        with self.assertRaises(ValueError):
            PrefixDataset(rows + [rows[0]])

    def test_forward_shape_vocabulary_and_prefix_mask_validation(self):
        model = PrefixTransformer()
        state = random_complete_state(random.Random(4))
        valid = torch.tensor([tokenize(state[:2]), tokenize(state)], dtype=torch.long)
        self.assertEqual(tuple(model(valid).shape), (2,))
        for bad in (
            torch.zeros((1, 6), dtype=torch.long),
            valid.float(),
            torch.full((1, 22), 18, dtype=torch.long),
        ):
            with self.assertRaises(ValueError):
                model(bad)
        bad = valid.clone()
        bad[0, 0] = UNASSIGNED
        with self.assertRaises(ValueError):
            model(bad)
        bad = valid.clone()
        bad[1, 0] = 7
        with self.assertRaises(ValueError):
            model(bad)

    def test_freeze_checkpoint_round_trip_and_cross_benchmark_rejection(self):
        torch.manual_seed(5)
        predictor = FrozenPredictor(PrefixTransformer(), 5, 2)
        state = random_complete_state(random.Random(5))
        prefixes = [(), state[:9], state]
        self.assertTrue(np.array_equal(predictor.predict(prefixes), predictor.predict(prefixes)))
        self.assertFalse(predictor.model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in predictor.model.parameters())
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            metadata = {"split_seed": 101, "model_seed": 0, "data_sha256": "a" * 64}
            predictor.save(path, metadata)
            loaded = FrozenPredictor.load(path)
            self.assertTrue(np.array_equal(loaded.predict(prefixes), predictor.predict(prefixes)))
            with self.assertRaises(ValueError):
                Predictor201.load(path)
            payload = json.loads(path.read_text())
            payload["encoding"] = "other"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                FrozenPredictor.load(path)

    def test_training_determinism_normalization_and_overlap(self):
        rows = records(8)
        config = TrainingConfig(epochs=2, patience=2, batch_size=64)
        first, report = fit(rows[:6], rows[6:], 9, config)
        second, _ = fit(rows[:6], rows[6:], 9, config)
        self.assertEqual(report["normalization"]["mean"], 3.5)
        prefixes = [tuple(row["state"][:3]) for row in rows]
        self.assertTrue(np.array_equal(first.predict(prefixes), second.predict(prefixes)))
        self.assertEqual(report["train_architectures"], 6)
        with self.assertRaises(ValueError):
            fit(rows[:6], rows[5:], 9, config)

    def test_evaluation_keeps_weights_and_gradients_unchanged(self):
        predictor = FrozenPredictor(PrefixTransformer(), 4, 2)
        before = {key: value.clone() for key, value in predictor.model.state_dict().items()}
        report = evaluate(predictor, records(5), batch_size=2)
        self.assertEqual(set(report["by_depth"]), {str(depth) for depth in range(1, 23)})
        self.assertTrue(
            all(
                torch.equal(before[key], value)
                for key, value in predictor.model.state_dict().items()
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in predictor.model.parameters()))


class OracleAndRunnerTests(unittest.TestCase):
    def test_worker_protocol_redirects_legacy_logs_and_disables_noise(self):
        state = random_complete_state(random.Random(14))
        genotype = state_to_genotype(state)
        surrogate = Mock()
        surrogate.predict.return_value = 94.0
        package = Mock()
        package.load_ensemble.return_value = surrogate
        request = {"genotype": genotype._asdict(), "with_noise": False}
        output = io.StringIO()
        with (
            patch.dict("sys.modules", {"nasbench301": package}),
            patch("sys.argv", ["oracle_bridge.py", "--model", "model-dir"]),
            patch("sys.stdin", io.StringIO(json.dumps(request) + "\n")),
            patch("sys.stdout", output),
        ):
            bridge_main()
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(responses, [{"ready": True}, {"accuracy": 94.0}])
        package.load_ensemble.assert_called_once_with("model-dir")
        self.assertFalse(surrogate.predict.call_args.kwargs["with_noise"])

    def test_noise_disabled_caching_and_prefix_rejection(self):
        surrogate = Mock()
        surrogate.predict.return_value = 93.5
        oracle = TerminalOracle(surrogate)
        state = random_complete_state(random.Random(3))
        self.assertEqual(oracle(state), 6.5)
        self.assertEqual(oracle(state), 6.5)
        surrogate.predict.assert_called_once_with(
            config=state_to_genotype(state), representation="genotype", with_noise=False
        )
        with self.assertRaises(ValueError):
            oracle(state[:-1])
        surrogate.predict.return_value = float("nan")
        oracle.clear()
        with self.assertRaises(ValueError):
            oracle(state)

    def test_missing_surrogate_has_precise_asset_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "surrogate_model.model"):
                TerminalOracle.load(directory)

    def test_search_metrics_have_no_global_optimum_fields(self):
        surrogate = Mock()
        surrogate.predict.return_value = 94.0
        oracle = TerminalOracle(surrogate)
        row = run_one(DEFAULT_SPECS[0], lambda state: float(state[-1]) if state else 0.0, oracle, 2)
        self.assertEqual(row["queries"], 2)
        self.assertEqual(row["max_depth"], 22)
        self.assertEqual(row["search_best_error"], 6.0)
        self.assertEqual(row["query_of_final_best"], 1)
        self.assertEqual(len(row["best_state"]), 22)
        self.assertNotIn("regret_pp", row)
        self.assertNotIn("optimum_found", row)
        self.assertGreater(row["python_peak_bytes"], 0)
        self.assertEqual(row["surrogate_calls"], 2)

    def test_random_reference_and_alignment_use_deterministic_terminal_oracle(self):
        surrogate = Mock()
        surrogate.predict.return_value = 94.0
        oracle = TerminalOracle(surrogate)
        first = make_reference(oracle, 5, 19)
        second = make_reference(oracle, 5, 19)
        self.assertEqual(first, second)
        self.assertEqual(first["size"], 5)
        report = surrogate_alignment(lambda state: 6.0, oracle, [()], completions=3)
        self.assertEqual(report["rows"][0]["completion_mean_error"], 6.0)

    def test_all_initial_specs_dispatch_through_the_shared_registry(self):
        tree = ArchitectureTree(lambda state: float(state[-1]) if state else 0, lambda state: 6.0)
        for spec in DEFAULT_SPECS:
            result = execute(spec, tree, Budget(queries=1))
            self.assertEqual(result.queries, 1, spec.name)


if __name__ == "__main__":
    unittest.main()
