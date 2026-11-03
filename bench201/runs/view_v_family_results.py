"""Edit the constants below, then press the IDE Run button."""

from _launch import launch

# ---------- EDITABLE VIEW SETTINGS ----------
RESULTS = "artifacts/v-family-absolute-ablation-001"
INITIAL_BUDGET = 100
INITIAL_FAMILY = "VCDFS-A"  # VCDFS, VCDFS-A, VCBFS, VCBFS-A
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
        INITIAL_FAMILY,
    )
