"""JSONL entry point for minimal official XGBoost inference in Python 3.8."""

import argparse
import contextlib
import json
import math
import random
import sys
from pathlib import Path

# The embedded interpreter intentionally ignores system/site paths.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench301.representation import Genotype, random_complete_state, state_to_genotype
from bench301.xgb_runtime import XGBEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = XGBEnsemble(args.model)
            if args.prepare:
                rng = random.Random(101)
                genotypes = [state_to_genotype(random_complete_state(rng)) for _ in range(3)]
                report = model.prepare_native(genotypes)
        if args.prepare:
            print(json.dumps(report, allow_nan=False), flush=True)
            return
        print(json.dumps({"ready": True}), flush=True)
    except Exception as error:
        print(json.dumps({"error": str(error)}), flush=True)
        raise SystemExit(1) from error
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("with_noise") is not False:
                raise ValueError("Noise must be disabled")
            with contextlib.redirect_stdout(sys.stderr):
                accuracy = model.predict(
                    config=Genotype(**request["genotype"]),
                    representation="genotype",
                    with_noise=False,
                )
            if not math.isfinite(accuracy):
                raise ValueError("Nonfinite prediction")
            response = {"accuracy": accuracy}
        except Exception as error:
            response = {"error": str(error)}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
