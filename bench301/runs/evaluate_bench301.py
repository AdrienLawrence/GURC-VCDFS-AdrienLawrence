"""Print held-out real-data diagnostics for an existing frozen NB301 model."""

from _launch import launch

DATA = "bench301/artifacts/measurements.json"
CHECKPOINT = "bench301/artifacts/full-study/split-101-model-0"

if __name__ == "__main__":
    launch("-m", "bench301.heuristic.evaluate", "--data", DATA, "--checkpoint", CHECKPOINT)
