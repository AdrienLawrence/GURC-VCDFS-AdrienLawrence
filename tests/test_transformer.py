"""ML correctness tests use synthetic labels; no empirical NAS claims."""

import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from heuristic.data import examples, split_indices, tokenize
from heuristic.evaluate import Baselines, metrics
from heuristic.extract import CompactMemo, StreamingReader, summarize
from heuristic.model import FrozenPredictor, PrefixTransformer
from heuristic.train import TrainingConfig, fit
from nas_space import architecture_string


def row(state, error):
    return {"state": list(state), "error": error}


class DataTests(unittest.TestCase):
    def test_architecture_split_is_disjoint_complete_reproducible(self):
        split = split_indices(15625, 101)
        self.assertEqual(split, split_indices(15625, 101))
        self.assertEqual([len(split[k]) for k in ("train", "dev", "test")], [1024, 256, 14345])
        combined = sum(split.values(), [])
        self.assertEqual(len(set(combined)), 15625)
        self.assertNotEqual(split, split_indices(15625, 203))

    def test_prefix_labels_and_equal_depth_weights(self):
        records = [row((0, 1, 2, 3, 4, 0), 8), row((0, 1, 2, 3, 4, 1), 12)]
        x, y, depths = examples(records)
        self.assertEqual(x.shape, (12, 6))
        np.testing.assert_array_equal(x[0], x[6])
        self.assertEqual((y[0] + y[6]) / 2, 10)
        np.testing.assert_array_equal(np.bincount(depths)[1:], [2] * 6)

    def test_grouped_weighted_mse_has_identical_gradient(self):
        outcomes = np.array([8, 12, 15])
        estimate = 11.0
        self.assertAlmostEqual(
            2 * np.sum(estimate - outcomes), 2 * len(outcomes) * (estimate - outcomes.mean())
        )

    def test_duplicate_architectures_and_invalid_labels_rejected(self):
        for records in ([row((0,) * 6, 8)] * 2, [row((0,) * 6, float("nan"))], []):
            with self.assertRaises(ValueError):
                examples(records)

    def test_lookup_falls_back_to_longest_observed_prefix(self):
        baseline = Baselines([row((0,) * 6, 8), row((1,) * 6, 12)])
        self.assertEqual(baseline.predict([tokenize((0, 4))], "prefix")[0], 8)
        self.assertEqual(baseline.predict([tokenize((4,))], "prefix")[0], 10)

    def test_rank_metrics_handle_ties_and_constant_predictions(self):
        self.assertEqual(metrics([3, 2, 1], [3, 2, 1])["spearman"], 1)
        self.assertIsNone(metrics([2, 2, 2], [1, 2, 3])["spearman"])


class ModelTests(unittest.TestCase):
    def test_architecture_parameter_count_and_batch_shape(self):
        model = PrefixTransformer()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 17633)
        self.assertEqual(tuple(model(torch.zeros(5, 6, dtype=torch.long)).shape), (5,))

    def test_tokenization_marks_only_unassigned_suffix(self):
        self.assertEqual(tokenize(()), (5, 5, 5, 5, 5, 5))
        self.assertEqual(tokenize((0, 4, 2)), (0, 4, 2, 5, 5, 5))

    def test_transformer_rejects_wrong_shape_dtype_and_token(self):
        model = PrefixTransformer()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 5, dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 6))
        with self.assertRaises(ValueError):
            model(torch.full((2, 6), 6, dtype=torch.long))

    def test_prefix_and_pretokenized_prediction_interfaces_agree(self):
        predictor = FrozenPredictor(PrefixTransformer(), 0, 1)
        prefixes = [(0, 1), (2, 3, 4)]
        tokens = np.asarray([tokenize(prefix) for prefix in prefixes])
        np.testing.assert_array_equal(predictor.predict(prefixes), predictor.predict_tokens(tokens))
        with self.assertRaises(ValueError):
            predictor.predict_tokens(np.zeros((2, 6), dtype=np.float32))

    def test_frozen_inference_disables_dropout(self):
        predictor = FrozenPredictor(PrefixTransformer(dropout=0.5), 0, 1)
        first = predictor.predict([(0, 1), (2, 3)])
        second = predictor.predict([(0, 1), (2, 3)])
        np.testing.assert_array_equal(first, second)

    def test_unassigned_operation_embedding_is_attention_masked(self):
        predictor = FrozenPredictor(PrefixTransformer(), 0, 1)
        prefix = (0, 1)
        before = predictor(prefix)
        with torch.no_grad():
            predictor.model.operation_embedding.weight[5].fill_(1_000_000)
        after = predictor(prefix)
        self.assertAlmostEqual(before, after, places=6)

    def test_checkpoint_roundtrip_and_frozen_inference(self):
        predictor = FrozenPredictor(PrefixTransformer(), 10, 3)
        x = [(1, 2), (3,)]
        expected = predictor.predict(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            predictor.save(path, {})
            actual = FrozenPredictor.load(path)
            np.testing.assert_array_equal(actual.predict(x), expected)
            self.assertFalse(actual.model.training)
            self.assertTrue(all(not p.requires_grad for p in actual.model.parameters()))
            with self.assertRaises(FileExistsError):
                predictor.save(path, {})

    def test_nonfinite_checkpoint_is_rejected(self):
        model = PrefixTransformer()
        with torch.no_grad():
            next(model.parameters())[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            FrozenPredictor(model, 0, 1)

    def test_training_rejects_architecture_overlap(self):
        records = [row((0,) * 6, 10)]
        with self.assertRaises(ValueError):
            fit(records, records)

    def test_training_deterministic_and_normalization_train_only(self):
        train = [row((i, 0, 0, 0, 0, 0), 8 + i) for i in range(4)]
        dev = [row((4, 0, 0, 0, 0, 0), 90)]
        config = TrainingConfig(epochs=4, patience=2)
        a, ra = fit(train, dev, seed=9, config=config)
        b, rb = fit(train, dev, seed=9, config=config)
        self.assertEqual(ra["normalization"]["mean"], 9.5)
        self.assertEqual(ra["history"], rb["history"])
        self.assertEqual(a((2,)), b((2,)))
        losses = [entry["dev_mse_pp2"] for entry in ra["history"]]
        self.assertEqual(ra["best_epoch"], losses.index(min(losses)) + 1)

    def test_small_mapping_can_be_learned(self):
        train = [row((i, j, 0, 0, 0, 0), 5 + 3 * i) for i in range(5) for j in range(4)]
        dev = [row((i, 4, 0, 0, 0, 0), 5 + 3 * i) for i in range(5)]
        model, _ = fit(train, dev, seed=1, config=TrainingConfig(epochs=100))
        predicted = model.predict([(i,) for i in range(5)])
        self.assertLess(
            float(np.abs(predicted - np.array([5 + 3 * i for i in range(5)])).mean()), 1
        )

    def test_bad_training_settings_fail(self):
        for kwargs in ({"epochs": 0}, {"patience": True}, {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs)


class ExtractionTests(unittest.TestCase):
    @staticmethod
    def record(index):
        return {
            "full": {
                "arch_index": index,
                "arch_str": architecture_string((index,) * 6),
                "all_results": {
                    ("cifar10-valid", 777): {
                        "epochs": 200,
                        "eval_acc1es": {"x-valid@199": 91.5, "ori-test@199": 99},
                    }
                },
            },
            "less": {},
        }

    def test_extracts_only_final_validation_not_test(self):
        result = summarize(self.record(0))
        self.assertEqual(result["error"], 8.5)
        self.assertNotIn("99", json.dumps(result))

    def test_streaming_matches_plain_pickle_on_two_records(self):
        records = {0: self.record(0), 1: self.record(1)}
        serialized = pickle.dumps(records, protocol=2)
        expected = [summarize(v) for v in pickle.loads(serialized).values()]
        actual = []
        StreamingReader(io.BytesIO(serialized), actual.append).load()
        self.assertEqual(actual, expected)

    def test_rejects_wrong_epoch_missing_metric_and_constructor(self):
        bad = self.record(0)
        next(iter(bad["full"]["all_results"].values()))["epochs"] = 12
        with self.assertRaises(ValueError):
            summarize(bad)
        with self.assertRaises(ValueError):
            StreamingReader(io.BytesIO(pickle.dumps(Path("x"), protocol=2)), lambda _: None).load()

    def test_discarded_container_reference_fails_closed(self):
        memo = CompactMemo()
        memo[0], memo[1] = {"large": []}, "shared"
        memo.release_record()
        self.assertEqual(memo[1], "shared")
        with self.assertRaises(ValueError):
            _ = memo[0]
