"""Independent recursive oracles and adversarial scheduler tests."""

import random
import unittest

from algs import (
    Budget,
    DynamicBeamConfig,
    Retention,
    beam,
    bfs,
    cdfs,
    dfs,
    dynamic_beam,
    gbfs,
    heuristic_dfs,
    search,
    vcbfs,
    vcbfsa,
    vcdfs,
    vcdfsa,
)
from run import DemoTree


class SearchProperties(unittest.TestCase):
    def test_random_trees_against_independent_recursive_oracle(self):
        for seed in range(30):
            rng = random.Random(seed)
            children, scores, objectives = {}, {}, {}

            def build(state):
                scores[state] = rng.randrange(5)
                width = rng.randrange(5) if len(state) < 4 else 0
                children[state] = [state + (i,) for i in range(width)]
                if not width:
                    objectives[state] = rng.randrange(100)
                for child in children[state]:
                    build(child)

            build(())

            class Tree:
                initial = ()
                feasible = staticmethod(lambda s: True)
                terminal = staticmethod(lambda s: not children[s])
                successors = staticmethod(lambda s: children[s])
                heuristic = staticmethod(lambda s: scores[s])
                objective = staticmethod(lambda s: objectives[s])

            for policy in (Retention("absolute", 1, 3), Retention("proportional", 20, 80)):
                terminals = []

                def visit(state):
                    if not children[state]:
                        terminals.append(state)
                        return
                    ranked = sorted(children[state], key=scores.get)
                    values = [scores[c] for c in ranked]
                    if len(values) == 1:
                        k = 1
                    else:
                        mean = sum(values) / len(values)
                        mad = sum(abs(v - mean) for v in values) / len(values)
                        conf = min(1, (values[1] - values[0]) / (mad + 1e-5))
                        width = policy.lower + (1 - conf) * (policy.upper - policy.lower)
                        if policy.mode == "proportional":
                            width *= len(values) / 100
                        k = min(len(values), max(1, int(width)))
                    for child in ranked[:k]:
                        visit(child)

                visit(())
                actual = []
                result = search(
                    Tree(),
                    policy,
                    event=lambda e: actual.append(e["state"]) if e["kind"] == "terminal" else None,
                )
                self.assertEqual(actual, terminals)
                self.assertEqual(result.best_value, min(objectives[s] for s in terminals))
            for algorithm in (dfs, bfs, heuristic_dfs, gbfs):
                self.assertEqual(algorithm(Tree()).best_value, min(objectives.values()))

    def test_public_algorithms_match_dispatcher(self):
        self.assertEqual(cdfs(DemoTree(), 2), search(DemoTree(), Retention("fixed", 2, 2)))
        self.assertEqual(
            vcdfs(DemoTree(), 20, 80), search(DemoTree(), Retention("proportional", 20, 80))
        )
        self.assertEqual(vcdfsa(DemoTree(), 1, 3), search(DemoTree(), Retention("absolute", 1, 3)))

    def test_blind_algorithms_never_call_heuristic(self):
        class Blind(DemoTree):
            def heuristic(self, state):
                raise AssertionError("Blind traversal must not score")

        for algorithm in (dfs, bfs):
            self.assertEqual(algorithm(Blind()).best_value, 0)

    def test_terminal_is_never_expanded_even_at_zero_expansion_budget(self):
        class Terminal(DemoTree):
            terminal = staticmethod(lambda s: True)
            objective = staticmethod(lambda s: 5)

            def successors(self, state):
                raise AssertionError("Terminal expanded")

        for algorithm in (dfs, bfs, beam, gbfs, vcbfs, vcbfsa):
            self.assertEqual(algorithm(Terminal(), budget=Budget(expansions=0)).best_value, 5)

    def test_nan_heuristic_and_invalid_width_rejected(self):
        class Bad(DemoTree):
            heuristic = staticmethod(lambda s: float("nan"))

        for algorithm in (heuristic_dfs, cdfs, vcdfs, beam, gbfs, vcbfs, vcbfsa):
            with self.assertRaises(ValueError):
                algorithm(Bad())
        for width in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                beam(DemoTree(), width)

    def test_all_schedulers_respect_query_cap(self):
        def dynamic(problem, **kwargs):
            return dynamic_beam(problem, DynamicBeamConfig("entropy", 1, 3, 2, 1), **kwargs)

        for algorithm in (dfs, bfs, beam, cdfs, vcdfs, vcdfsa, gbfs, vcbfs, vcbfsa, dynamic):
            for limit in (0, 1, 2, 100):
                result = algorithm(DemoTree(), budget=Budget(queries=limit))
                self.assertLessEqual(result.queries, limit)
