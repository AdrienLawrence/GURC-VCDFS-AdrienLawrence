"""Create a frozen random reference; its best architecture is not a global optimum."""

import argparse
import random
from pathlib import Path

from schemas import write_artifact

from .oracle import TerminalOracle
from .representation import ENCODING, random_complete_state


def make_reference(oracle, size, seed):
    if type(size) is not int or size < 1:
        raise ValueError("Reference size must be a positive integer")
    rng, seen, records = random.Random(seed), set(), []
    while len(records) < size:
        state = random_complete_state(rng)
        if state in seen:
            continue
        seen.add(state)
        records.append({"state": list(state), "error": oracle(state)})
    return {
        "benchmark": "nasbench301",
        "encoding": ENCODING,
        "scope": "uniform valid complete architectures; sample best is not a proven optimum",
        "seed": seed,
        "size": size,
        "oracle": oracle.provenance,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-model", type=Path, required=True)
    parser.add_argument("--oracle-python", type=Path)
    parser.add_argument("--oracle-backend", choices=("official", "minimal-xgb"), default="official")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Reference output already exists; choose a new file")
    oracle = TerminalOracle.load(args.oracle_model, args.oracle_python, backend=args.oracle_backend)
    try:
        payload = make_reference(oracle, args.size, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_artifact(args.output, "nb301_reference", payload)
        print(f"Saved {args.size} reference architectures -> {args.output}", flush=True)
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
