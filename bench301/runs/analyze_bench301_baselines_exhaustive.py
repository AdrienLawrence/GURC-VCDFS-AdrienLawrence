"""Run the safety-bounded NB301 exhaustive baseline study from the IDE."""

from _launch import launch

# ---------- EDITABLE RUN SETTINGS ----------
DATA = "bench301/artifacts/measurements.json"
STUDY = "bench301/artifacts/full-study"
OUTPUT = "bench301/artifacts/baseline-exhaustive-001"
MODELS_PER_SPLIT = 1
CHECKPOINT_EVERY = 1
ORACLE_MODEL = "bench301/cache/nb_models_1.0/xgb_v1.0"
ORACLE_PYTHON = "bench301/cache/oracle-runtime/python.exe"
ORACLE_BACKEND = "minimal-xgb"
MAX_SECONDS = 180
MAX_RSS_DELTA_MIB = 512
MAX_EXPANSIONS = 5_000
# -------------------------------------------

if __name__ == "__main__":
    launch(
        "-m",
        "bench301.behavior.experiment",
        "--data",
        DATA,
        "--study",
        STUDY,
        "--output",
        OUTPUT,
        "--preset",
        "baselines",
        "--exhaustive",
        "--models-per-split",
        MODELS_PER_SPLIT,
        "--checkpoint-every",
        CHECKPOINT_EVERY,
        "--oracle-model",
        ORACLE_MODEL,
        "--oracle-python",
        ORACLE_PYTHON,
        "--oracle-backend",
        ORACLE_BACKEND,
        "--max-seconds",
        MAX_SECONDS,
        "--max-rss-delta-mib",
        MAX_RSS_DELTA_MIB,
        "--max-expansions",
        MAX_EXPANSIONS,
    )
