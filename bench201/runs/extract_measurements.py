"""Edit the constants below, then press the IDE Run button."""

from _launch import launch

# ---------- EDITABLE RUN SETTINGS ----------
DATASET = "NAS-Bench-201-v1_1-096897.pth"
OUTPUT = "artifacts/measurements.json"
# -------------------------------------------

if __name__ == "__main__":
    launch("-m", "bench201.heuristic.extract", "--dataset", DATASET, "--output", OUTPUT)
