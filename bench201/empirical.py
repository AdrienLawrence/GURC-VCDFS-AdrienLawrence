"""Run frozen-heuristic NAS searches and write auditable empirical measurements."""

import argparse
import gc
import platform
import statistics
import time
import tracemalloc
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from algs import Budget
from bench201.heuristic.data import load_measurements
from bench201.heuristic.experiment import source_digest
from bench201.heuristic.model import FrozenPredictor, checkpoint_metadata
from bench201.space import ArchitectureTree, architecture_string
from bench201.specs import execute, load_specs
from resource_measurement import RssSampler
from schemas import load_payload, write_artifact
from search_reporting import (
    SCALAR_COLUMNS,
    json_histogram,
    print_final_summary,
    print_run,
    quality_metrics,
    save_csv,
    summarize,
)

# Human-editable output defaults. CLI --[no-] switches override these.
PRINT_RUNS = True
PRINT_SUMMARY = True
WRITE_JSON = True
WRITE_CSV = False


class EventSummary:
    def __init__(self):
        self.terminals = set()
        self.widths = Counter()
        self.confidences = []
        self.statistics = []

    def __call__(self, event):
        kind = event["kind"]
        if kind == "terminal":
            self.terminals.add(tuple(event["state"]))
        if kind in {"retain", "beam", "vcbfs_beam", "dynamic_beam"}:
            self.widths[int(event["k"])] += 1
        if kind == "vcbfs":
            self.widths.update(int(group["k"]) for group in event["groups"])
        if event.get("confidence") is not None:
            self.confidences.append(float(event["confidence"]))
        if event.get("statistic") is not None:
            self.statistics.append(float(event["statistic"]))


def _same_result(first, second):
    fields = (
        "best_state",
        "best_value",
        "stop",
        "expanded",
        "generated",
        "scored",
        "queries",
        "rejected",
        "pruned",
        "max_depth",
        "peak_pending",
    )
    return all(getattr(first, field) == getattr(second, field) for field in fields)


def measure_search(spec, problem_factory, budget):
    events = EventSummary()
    started = time.perf_counter()
    result = execute(spec, problem_factory(), budget, events)
    elapsed = time.perf_counter() - started

    # A separate deterministic replay avoids charging tracemalloc overhead to
    # runtime. This is Python allocation memory only, not model weights or RSS.
    gc.collect()
    tracemalloc.start()
    with RssSampler() as rss:
        memory_result = execute(spec, problem_factory(), budget)
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not _same_result(result, memory_result):
        raise RuntimeError(f"Nondeterministic replay for {spec.name}")
    rss_delta = None if rss.start is None else max(0, rss.peak - rss.start)
    return result, events, elapsed, python_peak_bytes, rss.peak, rss_delta


def run_checkpoint(
    directory, records, data_checksum, specs, query_budget, *, print_runs=PRINT_RUNS
):
    directory = Path(directory)
    metadata = checkpoint_metadata(directory / "model.json")
    if metadata["data_sha256"] != data_checksum:
        raise ValueError("Checkpoint and measurement data checksums differ")
    split = load_payload(directory / "split.json", "data_split", allow_legacy=True)
    flat = split["train"] + split["dev"] + split["test"]
    if len(flat) != len(records) or len(set(flat)) != len(records):
        raise ValueError("Checkpoint split is not a complete disjoint partition")
    predictor = FrozenPredictor.load(directory / "model.json")
    predictor(())  # deterministic one-call warmup outside measurements
    objectives = {tuple(row["state"]): float(row["error"]) for row in records}
    known = {tuple(records[index]["state"]) for index in split["train"] + split["dev"]}
    initial_incumbent = min(objectives[state] for state in known)
    oracle_minimum = min(objectives.values())

    def problem_factory():
        return ArchitectureTree(predictor, objectives.__getitem__)

    rows = []
    for spec in specs:
        print(f"Running {directory.name}: {spec.name}", flush=True)
        result, events, elapsed, python_peak, rss_peak, rss_delta = measure_search(
            spec, problem_factory, Budget(queries=query_budget)
        )
        quality = quality_metrics(result.best_value, initial_incumbent, oracle_minimum)
        row = {
            "checkpoint": directory.name,
            "split_seed": int(metadata["split_seed"]),
            "model_seed": int(metadata["model_seed"]),
            "algorithm": spec.name,
            "implementation": spec.algorithm,
            "parameters": spec.parameters,
            "query_budget": query_budget,
            "stop": result.stop,
            "runtime_seconds": elapsed,
            "python_peak_bytes": python_peak,
            "rss_peak_bytes": rss_peak,
            "rss_peak_delta_bytes": rss_delta,
            "peak_pending_entries": result.peak_pending,
            "expanded": result.expanded,
            "generated": result.generated,
            "scored": result.scored,
            "queries": result.queries,
            "unique_terminal_states": len(events.terminals),
            "new_terminal_states": len(events.terminals - known),
            "rejected": result.rejected,
            "pruned": result.pruned,
            "max_depth": result.max_depth,
            "search_best_error": result.best_value,
            "search_best_architecture": (
                architecture_string(result.best_state) if result.best_state is not None else None
            ),
            "initial_known_error": initial_incumbent,
            "oracle_minimum_error": oracle_minimum,
            **quality,
            "width_histogram": json_histogram(events.widths),
            "mean_confidence": (
                statistics.fmean(events.confidences) if events.confidences else None
            ),
            "confidence_saturation_fraction": (
                sum(value in {0.0, 1.0} for value in events.confidences) / len(events.confidences)
                if events.confidences
                else None
            ),
            "mean_controller_statistic": (
                statistics.fmean(events.statistics) if events.statistics else None
            ),
        }
        rows.append(row)
        if print_runs:
            print_run(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/measurements.json"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--print-runs", action=argparse.BooleanOptionalAction, default=PRINT_RUNS)
    parser.add_argument(
        "--print-summary", action=argparse.BooleanOptionalAction, default=PRINT_SUMMARY
    )
    parser.add_argument("--write-json", action=argparse.BooleanOptionalAction, default=WRITE_JSON)
    parser.add_argument("--write-csv", action=argparse.BooleanOptionalAction, default=WRITE_CSV)
    args = parser.parse_args()
    if args.queries < 1:
        parser.error("--queries must be positive")
    if (args.write_json or args.write_csv) and args.output is None:
        parser.error("--output is required when JSON or CSV writing is enabled")
    records, checksum = load_measurements(args.data)
    specs = load_specs(args.specs)
    checkpoints = sorted(path for path in args.study.glob("split-*-model-*") if path.is_dir())
    if not checkpoints:
        parser.error("--study contains no split-*-model-* checkpoint directories")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "scope": "FROZEN TRANSFORMER NAS SEARCH EXPERIMENT",
        "data_sha256": checksum,
        "source_sha256": source_digest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "query_budget": args.queries,
        "regret_definition": "search_best_error - full_table_oracle_minimum_error",
        "incumbent_diagnostic": (
            "min(initial_train_dev_incumbent, search_best_error) - full_table_oracle_minimum_error"
        ),
        "checkpoint_directories": [path.name for path in checkpoints],
        "algorithms": [asdict(spec) for spec in specs],
        "outputs": {
            "print_runs": args.print_runs,
            "print_summary": args.print_summary,
            "write_json": args.write_json,
            "write_csv": args.write_csv,
        },
        "memory_definition": (
            "separate deterministic replay: tracemalloc peak covers Python allocations; sampled "
            "RSS covers the whole process and delta is relative to the post-load pre-search "
            "baseline; peak_pending_entries is the algorithmic frontier proxy"
        ),
        "timing_definition": "perf_counter wall time after one predictor warmup; no tracemalloc",
        "independence_note": (
            "model seeds within one split are repeated fits, not independent datasets; final "
            "summary variance is computed across split means"
        ),
    }
    print(
        f"SEARCH STUDY | checkpoints={len(checkpoints)} algorithms={len(specs)} "
        f"query_budget={args.queries} write_json={args.write_json} write_csv={args.write_csv}",
        flush=True,
    )
    if args.write_json:
        write_artifact(args.output / "manifest.json", "search_manifest", manifest)
    rows = []
    for checkpoint in checkpoints:
        rows.extend(
            run_checkpoint(
                checkpoint,
                records,
                checksum,
                specs,
                args.queries,
                print_runs=args.print_runs,
            )
        )
    per_split, summary = summarize(rows)
    if args.print_summary:
        print_final_summary(summary)
    if args.write_json:
        write_artifact(args.output / "runs.json", "search_runs", rows)
        write_artifact(args.output / "split_summary.json", "search_split_summary", per_split)
        write_artifact(args.output / "summary.json", "search_summary", summary)
    if args.write_csv and per_split:
        save_csv(args.output / "runs.csv", rows, SCALAR_COLUMNS)
        save_csv(args.output / "split_summary.csv", per_split, tuple(per_split[0]))
    if args.write_csv and summary:
        save_csv(args.output / "summary.csv", summary, tuple(summary[0]))


if __name__ == "__main__":
    main()
