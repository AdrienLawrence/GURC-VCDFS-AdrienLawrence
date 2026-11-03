"""Extract verified real NB301 results; edit settings and press Run."""

from _launch import launch

SOURCE = "bench301/cache/nasbench301_full_data.zip"
OUTPUT = "bench301/artifacts/measurements.json"

if __name__ == "__main__":
    launch("-m", "bench301.heuristic.extract", "--source", SOURCE, "--output", OUTPUT)
