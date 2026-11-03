"""Run shared algorithms with a frozen prefix predictor and terminal oracle."""

from _launch import launch

CHECKPOINT = "bench301/artifacts/full-study/split-101-model-0"
ORACLE_MODEL = "bench301/cache/nb_models_1.0/xgb_v1.0"
# Set the independently prepared interpreter here, or set NB301_PYTHON.
ORACLE_PYTHON = "bench301/cache/oracle-runtime/python.exe"
ORACLE_BACKEND = "minimal-xgb"
QUERY_BUDGET = 8
SPECS_FILE = None  # Defaults are initial controls in bench301/specs.py.
REFERENCE = None
WRITE_JSON = False
OUTPUT = "bench301/artifacts/search-001"

if __name__ == "__main__":
    arguments = [
        "-m",
        "bench301.empirical",
        "--checkpoint",
        CHECKPOINT,
        "--oracle-model",
        ORACLE_MODEL,
        "--queries",
        QUERY_BUDGET,
        "--oracle-backend",
        ORACLE_BACKEND,
        "--write-json" if WRITE_JSON else "--no-write-json",
    ]
    if WRITE_JSON:
        arguments.extend(("--output", OUTPUT))
    for option, value in (
        ("--oracle-python", ORACLE_PYTHON),
        ("--specs", SPECS_FILE),
        ("--reference", REFERENCE),
    ):
        if value is not None:
            arguments.extend((option, value))
    launch(*arguments)
