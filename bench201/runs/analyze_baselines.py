"""Run matched baseline behavior experiments; edit settings, then press Run."""

from _launch import launch

# ---------- EDITABLE RUN SETTINGS ----------
DATA = "artifacts/measurements.json"
STUDY = "artifacts/full-study"
OUTPUT = "artifacts/baseline-behavior-001"
QUERY_BUDGETS = [1, 8, 32, 64, 100]
# One model per split matches the absolute ablation: 15 × 5 × 5 = 375 runs.
MODELS_PER_SPLIT = 1
CHECKPOINT_EVERY = 10
# -------------------------------------------

if __name__ == "__main__":
    arguments = [
        "-m",
        "bench201.behavior.experiment",
        "--data",
        DATA,
        "--study",
        STUDY,
        "--output",
        OUTPUT,
        "--preset",
        "baselines",
        "--extend-existing",
        "--budgets",
        *QUERY_BUDGETS,
        "--checkpoint-every",
        CHECKPOINT_EVERY,
    ]
    if MODELS_PER_SPLIT is not None:
        arguments.extend(("--models-per-split", MODELS_PER_SPLIT))
    launch(*arguments)
