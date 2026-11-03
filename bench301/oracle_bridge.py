"""Standalone stdlib JSONL boundary; executed by the separate NB301 environment."""

import argparse
import contextlib
import json
import math
import sys
from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        # Legacy libraries print during load/query; keep stdout as JSONL only.
        with contextlib.redirect_stdout(sys.stderr):
            import nasbench301

            surrogate = nasbench301.load_ensemble(args.model)
        print(json.dumps({"ready": True}), flush=True)
    except Exception as error:
        print(json.dumps({"error": str(error)}), flush=True)
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("with_noise") is not False:
                raise ValueError("Main experiments require with_noise=False")
            genotype = Genotype(**request["genotype"])
            with contextlib.redirect_stdout(sys.stderr):
                value = surrogate.predict(
                    config=genotype, representation="genotype", with_noise=False
                )
            accuracy = float(value.item() if hasattr(value, "item") else value)
            if not math.isfinite(accuracy):
                raise ValueError("Nonfinite surrogate output")
            response = {"accuracy": accuracy}
        except Exception as error:
            response = {"error": str(error)}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
