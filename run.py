"""Read real benchmark metadata or demonstrate search on a labeled toy tree."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from algs import Budget, Retention, search
from nas_space import EDGES, FILENAME, OPERATIONS, parse_architecture, read_metadata


class DemoTree:
    """Hand-written mechanism example, NOT NAS measurements or research evidence."""

    initial = ()
    costs = {
        (0, 0): 8,
        (0, 1): 7,
        (0, 2): 6,
        (1, 0): 5,
        (1, 1): 4,
        (1, 2): 3,
        (2, 0): 2,
        (2, 1): 1,
        (2, 2): 0,
    }

    def feasible(self, state):
        return True

    def terminal(self, state):
        return len(state) == 2

    def successors(self, state):
        return [state + (x,) for x in range(3)]

    def heuristic(self, state):
        return float(state[-1])  # Deliberately prefers the wrong branch.

    def objective(self, state):
        return self.costs[state]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read actual dataset architecture metadata")
    inspect.add_argument("--dataset", type=Path, default=Path(__file__).with_name(FILENAME))
    demo = commands.add_parser("demo", help="Synthetic correctness example; not NAS results")
    demo.add_argument("--trace", action="store_true")
    demo.add_argument("--queries", type=int)
    args = parser.parse_args()
    if args.command == "inspect":
        records = read_metadata(args.dataset)
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "architectures": len(records),
                    "edges_in_encoding_order": EDGES,
                    "operations": OPERATIONS,
                    "first_architecture": records[0],
                    "first_operation_ids": parse_architecture(records[0]),
                    "full_tree_depth": 6,
                    "full_tree_max_branching": 5,
                    "full_tree_nodes": sum(5**d for d in range(7)),
                    "measurement_records_loaded": False,
                    "predictor_trained": False,
                },
                indent=2,
            )
        )
        return
    print("SYNTHETIC DEMONSTRATION: true optimum = 0; heuristic deliberately misleading.")
    variants = [
        ("DFS", {}),
        ("BFS", {"schedule": "bfs"}),
        ("DFS-H", {"policy": Retention("all")}),
        ("CDFS-2", {"policy": Retention("fixed", 2, 2)}),
        ("VCDFS-20-100pct", {"policy": Retention("proportional", 20, 100)}),
        ("VCDFS-A-1-3", {"policy": Retention("absolute", 1, 3)}),
        ("Beam-2", {"schedule": "beam", "beam_width": 2}),
    ]
    for name, settings in variants:
        print(name)
        result = search(
            DemoTree(),
            **settings,
            budget=Budget(queries=args.queries),
            event=(lambda value: print(json.dumps(value))) if args.trace else None,
        )
        print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
