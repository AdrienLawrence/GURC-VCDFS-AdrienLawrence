"""Launch the read-only Pareto result viewer."""

import argparse
from pathlib import Path

from .pareto import RESOURCE_METRICS, load_summary, show_pareto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/full-search-next"),
        help="Search output directory or its summary.json file",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(RESOURCE_METRICS),
        default="cost",
        help="Resource axis displayed when the window opens",
    )
    args = parser.parse_args()
    show_pareto(load_summary(args.results), args.metric)


if __name__ == "__main__":
    main()
