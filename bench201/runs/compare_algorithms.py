"""Edit the constants below, then press the IDE Run button."""

from _launch import launch

# ---------- EDITABLE RUN SETTINGS ----------
DATA = "artifacts/measurements.json"
STUDY = "artifacts/full-study"
OUTPUT = "artifacts/full-search-next"
QUERY_BUDGET = 100
SPECS_FILE = None  # Example: "my_algorithm_specs.json"
PRINT_EACH_RUN = True
PRINT_SUMMARY = True
WRITE_JSON = True
WRITE_CSV = True
# -------------------------------------------

if __name__ == "__main__":
    arguments = [
        "-m",
        "bench201.empirical",
        "--data",
        DATA,
        "--study",
        STUDY,
        "--output",
        OUTPUT,
        "--queries",
        QUERY_BUDGET,
        "--print-runs" if PRINT_EACH_RUN else "--no-print-runs",
        "--print-summary" if PRINT_SUMMARY else "--no-print-summary",
        "--write-json" if WRITE_JSON else "--no-write-json",
        "--write-csv" if WRITE_CSV else "--no-write-csv",
    ]
    if SPECS_FILE is not None:
        arguments.extend(("--specs", SPECS_FILE))
    launch(*arguments)
