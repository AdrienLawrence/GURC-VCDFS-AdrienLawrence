"""Open the matched baseline behavior artifact; edit settings, then press Run."""

from _launch import launch

# ---------- EDITABLE VIEW SETTINGS ----------
RESULTS = "artifacts/baseline-behavior-001"
INITIAL_BUDGET = 100
# --------------------------------------------

if __name__ == "__main__":
    launch(
        "-m",
        "ui.behavior_cli",
        "--results",
        RESULTS,
        "--budget",
        INITIAL_BUDGET,
        "--family",
        "Baselines",
    )
