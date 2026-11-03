"""Search-tree explanatory figure data-selection tests."""

import unittest

from bench201.behavior.tree_trace import (
    Decision,
    decisions_from_events,
    select_contrasting_decisions,
)


class TreeFigureTests(unittest.TestCase):
    def test_retain_event_preserves_real_sorted_scores_and_membership(self):
        event = {
            "kind": "retain",
            "state": (),
            "confidence": 0.75,
            "retained": [{"state": (1,), "score": 1.0}],
            "pruned": [
                {"state": (0,), "score": 3.0},
                {"state": (2,), "score": 2.0},
            ],
        }
        decision = decisions_from_events([event])[0]
        self.assertEqual(decision.parent, ())
        self.assertEqual([score for score, _ in decision.candidates], [1.0, 2.0, 3.0])
        self.assertEqual(decision.retained[0][1], (1,))

    def test_contrasting_path_decisions_prefer_separated_depths(self):
        def decision(depth, confidence):
            parent = (0,) * depth
            child = parent + (0,)
            return Decision(parent, confidence, ((1.0, child),), ((1.0, child),), (), (1.0, 2.0))

        low, high = select_contrasting_decisions(
            [decision(0, 0.1), decision(1, 0.95), decision(4, 0.8)],
        )
        self.assertEqual((len(low.parent), len(high.parent)), (0, 4))


if __name__ == "__main__":
    unittest.main()
