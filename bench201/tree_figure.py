"""Capture and display one real-data V-family search-tree explanation."""

import argparse
from pathlib import Path

from bench201.behavior.tree_trace import capture_figure_data
from ui.search_tree_figure import show_search_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/measurements.json"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/full-study/split-101-model-0")
    )
    parser.add_argument(
        "--algorithm", choices=("vcdfs", "vcdfsa", "vcbfs", "vcbfsa"), default="vcdfs"
    )
    parser.add_argument("--lower", type=float, default=20)
    parser.add_argument("--upper", type=float, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--queries", type=int, default=32)
    args = parser.parse_args()
    show_search_tree(
        capture_figure_data(
            args.data,
            args.checkpoint,
            args.algorithm,
            args.lower,
            args.upper,
            args.epsilon,
            args.queries,
        )
    )


if __name__ == "__main__":
    main()
