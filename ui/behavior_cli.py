"""Launch the read-only V-family behavioral result viewer."""

import argparse
from pathlib import Path

from .behavior import FAMILIES, load_behavior_summary, show_behavior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("artifacts/v-family-behavior-full-001")
    )
    parser.add_argument("--budget", type=int)
    parser.add_argument("--family", choices=tuple(FAMILIES), default="VCDFS")
    args = parser.parse_args()
    show_behavior(load_behavior_summary(args.results), args.budget, args.family)


if __name__ == "__main__":
    main()
