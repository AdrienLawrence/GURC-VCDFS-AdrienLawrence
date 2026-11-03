"""Create an optional frozen random reference using the deterministic terminal oracle."""

from _launch import launch

ORACLE_MODEL = "bench301/cache/nb_models_1.0/xgb_v1.0"
ORACLE_PYTHON = "bench301/cache/oracle-runtime/python.exe"
ORACLE_BACKEND = "minimal-xgb"
SIZE = 1000
SEED = 101
OUTPUT = "bench301/artifacts/random-reference-001.json"

if __name__ == "__main__":
    arguments = [
        "-m",
        "bench301.reference",
        "--oracle-model",
        ORACLE_MODEL,
        "--size",
        SIZE,
        "--seed",
        SEED,
        "--output",
        OUTPUT,
        "--oracle-backend",
        ORACLE_BACKEND,
    ]
    if ORACLE_PYTHON is not None:
        arguments.extend(("--oracle-python", ORACLE_PYTHON))
    launch(*arguments)
