"""Edit the constants below, then press the IDE Run button."""

from _launch import launch

# ---------- EDITABLE RUN SETTINGS ----------
DATA = "artifacts/measurements.json"
STUDY = "artifacts/full-study"
OUTPUT = "artifacts/v-family-absolute-ablation-001"
# quick = 18 configs; full = 90 configs; absolute-ablation = 45 A-family configs.
PRESET = "absolute-ablation"
QUERY_BUDGETS = [1, 8, 32, 64, 100]
# One model from each split gives 5 checkpoints and 45 × 5 × 5 = 1,125 runs.
# Use None to preserve every model checkpoint, as quick/full did previously.
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
        PRESET,
        "--extend-existing",
        "--budgets",
        *QUERY_BUDGETS,
        "--checkpoint-every",
        CHECKPOINT_EVERY,
    ]
    if MODELS_PER_SPLIT is not None:
        arguments.extend(("--models-per-split", MODELS_PER_SPLIT))
    launch(*arguments)
