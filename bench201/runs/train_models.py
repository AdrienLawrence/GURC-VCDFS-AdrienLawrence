"""Edit the constants below, then press the IDE Run button."""

from _launch import launch

# ---------- EDITABLE RUN SETTINGS ----------
DATA = "artifacts/measurements.json"
OUTPUT = "artifacts/full-study-next"
SPLIT_SEEDS = [101, 203, 307, 409, 503]
MODEL_SEEDS = [0, 1, 2]
EPOCHS = 300
# -------------------------------------------

if __name__ == "__main__":
    launch(
        "-m",
        "bench201.heuristic.experiment",
        "--data",
        DATA,
        "--output",
        OUTPUT,
        "--split-seeds",
        *SPLIT_SEEDS,
        "--model-seeds",
        *MODEL_SEEDS,
        "--epochs",
        EPOCHS,
    )
