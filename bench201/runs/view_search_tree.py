"""Edit the constants below, then press the IDE Run button."""

from _launch import launch

# ---------- EDITABLE FIGURE SETTINGS ----------
DATA = "artifacts/measurements.json"
CHECKPOINT = "artifacts/full-study/split-101-model-0"
ALGORITHM = "vcdfs"  # vcdfs, vcdfsa, vcbfs, vcbfsa
LOWER = 20  # p_min for proportional; k_min for absolute
UPPER = 100  # p_max for proportional; k_max for absolute
EPSILON = 1e-5
QUERY_BUDGET = 32
# ----------------------------------------------

if __name__ == "__main__":
    launch(
        "tree_figure.py",
        "--data",
        DATA,
        "--checkpoint",
        CHECKPOINT,
        "--algorithm",
        ALGORITHM,
        "--lower",
        LOWER,
        "--upper",
        UPPER,
        "--epsilon",
        EPSILON,
        "--queries",
        QUERY_BUDGET,
    )
