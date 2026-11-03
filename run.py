"""Read real benchmark metadata or demonstrate search on a labeled toy tree."""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from algs import Budget, Retention, search
from bench201.space import EDGES, FILENAME, OPERATIONS, parse_architecture, read_metadata


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
    commands.add_parser("test", help="Run the complete repository test suite")
    search_command = commands.add_parser(
        "search", help="Run and save the full 15-model, 100-query search comparison"
    )
    search_command.add_argument("--study", type=Path, default=Path("artifacts/full-study"))
    search_command.add_argument("--output", type=Path, default=Path("artifacts/full-search-001"))
    commands.add_parser("smoke", help="Run a one-query non-scientific search plumbing check")
    commands.add_parser("pilot", help="Train split 101/model 0 for up to 500 epochs")
    commands.add_parser("full", help="Train the preset five-split by three-model study")
    behavior = commands.add_parser(
        "behavior", help="Run the separate held-out NAS-Bench-201 V-family behavior suite"
    )
    behavior.add_argument("--study", type=Path, default=Path("artifacts/full-study"))
    behavior.add_argument("--output", type=Path, default=Path("artifacts/v-family-behavior-001"))
    behavior.add_argument(
        "--preset",
        choices=("quick", "full", "absolute-ablation", "baselines"),
        default="quick",
    )
    behavior.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 100])
    behavior.add_argument("--models-per-split", type=int)
    ui = commands.add_parser("ui", help="Open a read-only Pareto view of saved search results")
    ui.add_argument("--results", type=Path, default=Path("artifacts/full-search-001"))
    ui.add_argument(
        "--metric",
        choices=("cost", "runtime", "rss", "queries", "scores", "expansions", "pending"),
        default="cost",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == "test":
        subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=root,
            check=True,
        )
        return
    if args.command in {"search", "smoke"}:
        command = [
            sys.executable,
            "-B",
            "-m",
            "bench201.empirical",
            "--study",
            str(args.study) if args.command == "search" else "artifacts/pilot-study-long",
        ]
        if args.command == "smoke":
            command.extend(
                [
                    "--queries",
                    "1",
                    "--no-write-json",
                    "--no-write-csv",
                    "--no-print-runs",
                    "--no-print-summary",
                ]
            )
            print(
                "NON-SCIENTIFIC SMOKE RUN: query budget is intentionally reduced to 1.",
                flush=True,
            )
        else:
            command.extend(["--output", str(args.output), "--write-json", "--write-csv"])
        subprocess.run(command, cwd=root, check=True)
        return
    if args.command == "behavior":
        command = [
            sys.executable,
            "-B",
            "-m",
            "bench201.behavior.experiment",
            "--study",
            str(args.study),
            "--output",
            str(args.output),
            "--preset",
            args.preset,
            "--budgets",
            *(str(value) for value in args.budgets),
        ]
        if args.models_per_split is not None:
            command.extend(["--models-per-split", str(args.models_per_split)])
        subprocess.run(command, cwd=root, check=True)
        return
    if args.command in {"pilot", "full"}:
        command = [sys.executable, "-B", "-m", "bench201.heuristic.experiment"]
        if args.command == "pilot":
            command.extend(
                [
                    "--output",
                    "artifacts/pilot-study-next",
                    "--split-seeds",
                    "101",
                    "--model-seeds",
                    "0",
                    "--epochs",
                    "500",
                ]
            )
        else:
            command.extend(["--output", "artifacts/full-study"])
        subprocess.run(command, cwd=root, check=True)
        return
    if args.command == "ui":
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "ui",
                "--results",
                str(args.results),
                "--metric",
                args.metric,
            ],
            cwd=root,
            check=True,
        )
        return
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
