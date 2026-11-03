"""Exhaust the retained search tree for every absolute V-family configuration."""

from _launch import launch

# ---------- EDITABLE RUN SETTINGS ----------
DATA = "artifacts/measurements.json"
STUDY = "artifacts/full-study"
OUTPUT = "artifacts/v-family-absolute-exhaustive-001"
# One model from each split: 45 configurations × 5 checkpoints = 225 runs.
MODELS_PER_SPLIT = 1
CHECKPOINT_EVERY = 5
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
        "absolute-ablation",
        "--extend-existing",
        "--exhaustive",
        "--checkpoint-every",
        CHECKPOINT_EVERY,
    ]
    if MODELS_PER_SPLIT is not None:
        arguments.extend(("--models-per-split", MODELS_PER_SPLIT))
    launch(*arguments)
