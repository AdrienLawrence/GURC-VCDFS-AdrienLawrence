"""Exact small-tree assertions; no trained model or benchmark scores required."""

import itertools
import math
import unittest

from algs import (
    Budget,
    DynamicBeamConfig,
    Retention,
    VCBFSConfig,
    breadth_confidence,
    confidence,
    dynamic_beam,
    gbfs,
    search,
    vcbfs,
)
from algs.dynamic_beam import (
    costs_to_probabilities,
    shannon_entropy,
    top_k_standard_deviation,
)
from nas_space import ArchitectureTree, architecture_string, encode, parse_architecture
from run import DemoTree


class ConfidenceTests(unittest.TestCase):
    def test_mean_absolute_deviation_and_gap(self):
        scores = [1, 2, 3]
        expected = min(1, 1 / (2 / 3 + 1e-5))
        self.assertEqual(confidence(scores), expected)
        self.assertAlmostEqual(confidence([0, 1, 2, 3, 4]), 1 / (1.2 + 1e-5))

    def test_ties_and_degenerate_sets(self):
        for scores in ([5, 5, 5], [0, 0, 100]):
            self.assertEqual(confidence(scores), 0)
        self.assertIsNone(confidence([]))
        self.assertIsNone(confidence([1]))
        self.assertEqual(Retention().width([]), (0, None))
        self.assertEqual(Retention().width([1]), (1, None))

    def test_percentage_collapse(self):
        for scores in ([0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 2, 3, 4]):
            self.assertEqual(Retention("proportional", 1, 5).width(scores)[0], 1)

    def test_saturation_and_width_bounds(self):
        self.assertEqual(confidence([0, 10]), 1)
        self.assertEqual(Retention("absolute", 2, 4).width([0, 10, 10, 10])[0], 2)
        self.assertEqual(Retention("absolute", 2, 4).width([1] * 5)[0], 4)
        self.assertEqual(Retention("absolute", 10, 20).width([1] * 5)[0], 5)
        self.assertEqual(Retention("proportional", 100, 100).width([0, 1, 2])[0], 3)

    def test_affine_invariance_when_epsilon_scaled(self):
        values = [0, 1, 2, 3, 4]
        self.assertAlmostEqual(confidence(values), confidence([10 * x + 7 for x in values], 1e-4))

    def test_validation(self):
        for scores in ([math.nan, 1], [0, math.inf]):
            with self.assertRaises(ValueError):
                confidence(scores)
        for args in (
            ("absolute", 0, 2),
            ("proportional", 0, 101),
            ("fixed", 1, 2),
            ("absolute", 2, 1),
            ("absolute", 1.5, 3),
        ):
            with self.assertRaises(ValueError):
                Retention(*args)
        with self.assertRaises(ValueError):
            Budget(queries=-1)


class SearchTests(unittest.TestCase):
    def test_backtracks_and_exhausts_all_retained_children(self):
        terminals = []
        result = search(
            DemoTree(),
            Retention("fixed", 2, 2),
            event=lambda e: terminals.append(e["state"]) if e["kind"] == "terminal" else None,
        )
        self.assertEqual(terminals, [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual((result.expanded, result.generated, result.scored), (3, 9, 9))
        self.assertEqual((result.queries, result.pruned, result.best_value), (4, 3, 4))
        self.assertEqual(result.stop, "exhausted")

    def test_full_retention_matches_independent_enumeration(self):
        expected = min(DemoTree.costs.values())
        settings = (
            {},
            {"schedule": "bfs"},
            {"policy": Retention("all")},
            {"policy": Retention("proportional", 100, 100)},
            {"schedule": "beam", "beam_width": 9},
        )
        for kwargs in settings:
            with self.subTest(kwargs=kwargs):
                result = search(DemoTree(), **kwargs)
                self.assertEqual(result.best_value, expected)
                self.assertEqual(result.queries, 9)
                self.assertEqual(result.generated, 12)

    def test_pruning_can_lose_global_optimum(self):
        result = search(DemoTree(), Retention("absolute", 1, 3))
        self.assertEqual(result.queries, 1)
        self.assertEqual(result.best_value, 8)
        self.assertGreater(result.best_value, min(DemoTree.costs.values()))

    def test_root_terminal_and_dead_end(self):
        class RootTerminal(DemoTree):
            def terminal(self, state):
                return True

            def objective(self, state):
                return 7

        result = search(RootTerminal())
        self.assertEqual((result.best_value, result.expanded, result.queries), (7, 0, 1))

        class DeadEnd(DemoTree):
            def successors(self, state):
                return []

        result = search(DeadEnd(), Retention())
        self.assertIsNone(result.best_value)
        self.assertEqual((result.expanded, result.scored, result.queries), (1, 0, 0))

    def test_infeasible_root_and_children(self):
        class Infeasible(DemoTree):
            def feasible(self, state):
                return False

        self.assertEqual(search(Infeasible()).expanded, 0)

        class Filtered(DemoTree):
            def feasible(self, state):
                return 2 not in state

        result = search(Filtered(), Retention("all"))
        self.assertEqual(result.queries, 4)
        self.assertEqual(result.rejected, 3)
        self.assertEqual(result.scored, 6)

    def test_stable_ties(self):
        class Tied(DemoTree):
            def heuristic(self, state):
                return 0

        result = search(Tied(), Retention("fixed", 1, 1))
        self.assertEqual(result.best_state, (0, 0))

    def test_exact_budget_semantics(self):
        result = search(DemoTree(), budget=Budget(queries=2))
        self.assertEqual((result.queries, result.best_value, result.stop), (2, 7, "query_budget"))
        result = search(DemoTree(), budget=Budget(expansions=0))
        self.assertEqual((result.expanded, result.queries), (0, 0))
        result = search(DemoTree(), budget=Budget(expansions=2))
        self.assertEqual((result.expanded, result.queries, result.best_value), (2, 3, 6))
        self.assertEqual(search(DemoTree(), budget=Budget(queries=0)).queries, 0)

    def test_iterative_deep_single_child_tree(self):
        class Chain:
            initial = 0
            feasible = staticmethod(lambda s: True)
            terminal = staticmethod(lambda s: s == 2000)
            successors = staticmethod(lambda s: [s + 1])
            heuristic = staticmethod(lambda s: 0)
            objective = staticmethod(lambda s: s)

        result = search(Chain(), Retention())
        self.assertEqual((result.max_depth, result.best_value), (2000, 2000))

    def test_beam_is_global_level_retention(self):
        result = search(DemoTree(), schedule="beam", beam_width=2)
        self.assertEqual(result.queries, 2)  # Not two children per parent.
        self.assertEqual(result.generated, 9)


class DynamicBeamTests(unittest.TestCase):
    def test_probability_adapter_is_normalized_and_translation_invariant(self):
        first = costs_to_probabilities([1, 2, 3], 2)
        second = costs_to_probabilities([101, 102, 103], 2)
        self.assertAlmostEqual(sum(first), 1)
        self.assertEqual(first, second)
        self.assertGreater(first[0], first[1])

    def test_published_statistics(self):
        probabilities = [0.5, 0.3, 0.2]
        expected_entropy = -sum(p * math.log(p) for p in probabilities)
        self.assertAlmostEqual(shannon_entropy(probabilities), expected_entropy)
        self.assertAlmostEqual(top_k_standard_deviation(probabilities, 2), 0.1)

    def test_linear_mapping_clamps_and_rounds_half_up(self):
        entropy = DynamicBeamConfig("entropy", 1, 5, 0, 2.5)
        self.assertEqual(entropy.width([0, 0, 0, 0])[0], 3)
        low = DynamicBeamConfig("entropy", 2, 4, 0, -100)
        high = DynamicBeamConfig("entropy", 2, 4, 0, 100)
        self.assertEqual(low.width([0] * 10)[0], 2)
        self.assertEqual(high.width([0] * 10)[0], 4)

    def test_entropy_and_standard_deviation_dynamic_beams(self):
        settings = (
            DynamicBeamConfig("entropy", 1, 3, 2, 1, temperature=1),
            DynamicBeamConfig("standard_deviation", 1, 3, -10, 3, top_k=3),
        )
        for config in settings:
            events = []
            result = dynamic_beam(DemoTree(), config, event=events.append)
            self.assertGreater(result.queries, 0)
            self.assertTrue(any(event["kind"] == "dynamic_beam" for event in events))
            self.assertLessEqual(result.peak_pending, 9)

    def test_invalid_dynamic_beam_settings(self):
        bad = (
            ("unknown", 1, 2, 1, 0),
            ("entropy", 0, 2, 1, 0),
            ("entropy", 3, 2, 1, 0),
            ("entropy", 1, 2, math.nan, 0),
            ("standard_deviation", 1, 2, -1, 2),
        )
        for args in bad:
            with self.assertRaises(ValueError):
                DynamicBeamConfig(*args)
        with self.assertRaises(ValueError):
            DynamicBeamConfig("entropy", 1, 2, 1, 0, top_k=2)
        with self.assertRaises(ValueError):
            costs_to_probabilities([1], 0)


class BreadthBaselinesTests(unittest.TestCase):
    def test_gbfs_expands_global_best_without_pruning(self):
        visits = []
        result = gbfs(
            DemoTree(),
            event=lambda event: visits.append(event["state"]) if event["kind"] == "visit" else None,
        )
        self.assertEqual(visits[:4], [(), (0,), (0, 0), (1,)])
        self.assertEqual(result.pruned, 0)
        self.assertEqual(result.queries, 9)
        self.assertEqual(result.best_value, 0)

    def test_vcbfs_reference_confidence_and_rounding(self):
        self.assertEqual(breadth_confidence([7]), 1)
        self.assertEqual(breadth_confidence([7, 7, 7]), 0.5)
        self.assertEqual(breadth_confidence([0, 10]), 1)
        config = VCBFSConfig(10, 30, max_beam=64)
        self.assertEqual(config.width(100, 0), 30)
        self.assertEqual(config.width(100, 1), 10)
        self.assertEqual(VCBFSConfig(25, 25).width(10, 0), 3)

    def test_vcbfs_is_global_level_retention(self):
        events = []
        result = vcbfs(DemoTree(), 50, 50, event=events.append)
        selections = [event for event in events if event["kind"] == "vcbfs"]
        self.assertEqual((selections[0]["b"], selections[0]["k"]), (3, 2))
        self.assertEqual((selections[1]["b"], selections[1]["k"]), (6, 3))
        self.assertEqual(result.queries, 3)

    def test_vcbfs_validation(self):
        for args in ((-1, 10), (20, 10), (1, 101)):
            with self.assertRaises(ValueError):
                VCBFSConfig(*args)
        with self.assertRaises(ValueError):
            VCBFSConfig(1, 3, max_beam=0)
        with self.assertRaises(ValueError):
            breadth_confidence([])


class ArchitectureTests(unittest.TestCase):
    def test_all_architectures_round_trip(self):
        states = itertools.product(range(5), repeat=6)
        for state in states:
            self.assertEqual(parse_architecture(architecture_string(state)), state)

    def test_encoding(self):
        encoded = encode((3, 1))
        self.assertEqual(len(encoded), 36)
        self.assertEqual(sum(encoded), 6)
        self.assertEqual(encoded[3], 1)
        self.assertEqual(encoded[7], 1)
        self.assertEqual(encoded[17], 1)
        for value in ((5,), (True,), (0,) * 7):
            with self.assertRaises(ValueError):
                encode(value)

    def test_full_tree_counts_with_synthetic_evaluator(self):
        domain = ArchitectureTree(lambda x: 0, lambda s: sum(s))
        result = search(domain)
        self.assertEqual((result.expanded, result.queries, result.generated), (3906, 15625, 19530))
        self.assertEqual(result.best_value, 0)

    def test_heldout_trie(self):
        allowed = ((0,) * 6, (1,) * 6)
        domain = ArchitectureTree(lambda x: 0, lambda s: sum(s), allowed=allowed)
        result = search(domain, Retention("all"))
        self.assertEqual(result.queries, 2)
        self.assertEqual(result.expanded, 11)

    def test_architecture_tree_passes_prefix_to_predictor(self):
        observed = []
        domain = ArchitectureTree(lambda prefix: observed.append(prefix) or 0, lambda state: 0)
        self.assertEqual(domain.heuristic((1, 2)), 0)
        self.assertEqual(observed, [(1, 2)])

    def test_parser_rejects_invalid_edges(self):
        with self.assertRaises(ValueError):
            parse_architecture("|none~1|+|none~0|none~1|+|none~0|none~1|none~2|")


if __name__ == "__main__":
    unittest.main()
