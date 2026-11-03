"""Edit the constants below, then press the IDE Run button."""

from _launch import launch

# ---------- EDITABLE VIEW SETTINGS ----------
RESULTS = "artifacts/full-search-next"
METRIC = "cost"  # cost, runtime, rss, queries, scores, expansions, pending
# --------------------------------------------

if __name__ == "__main__":
    launch("-m", "ui", "--results", RESULTS, "--metric", METRIC)
