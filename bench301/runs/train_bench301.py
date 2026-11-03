"""Train NB301 prefix models; all labels come from real measured records."""

from _launch import launch

DATA = "bench301/artifacts/measurements.json"
OUTPUT = "bench301/artifacts/full-study"
SPLIT_SEEDS = [101, 203, 307, 409, 503]
MODEL_SEEDS = [0]
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 128
# None uses the complete architecture set with the declared 80/10/10 split.
TRAIN_SIZE = None
DEV_SIZE = None

if __name__ == "__main__":
    arguments = [
        "-m",
        "bench301.heuristic.experiment",
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
        "--patience",
        PATIENCE,
        "--batch-size",
        BATCH_SIZE,
    ]
    if TRAIN_SIZE is not None:
        arguments.extend(("--train-size", TRAIN_SIZE))
    if DEV_SIZE is not None:
        arguments.extend(("--dev-size", DEV_SIZE))
    launch(*arguments)
