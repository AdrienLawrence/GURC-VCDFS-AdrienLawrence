"""Run frozen-heuristic NAS searches and write auditable empirical measurements."""

import argparse
import csv
import ctypes
import gc
import json
import platform
import statistics
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Thread

import numpy as np
import torch

from algs import (
    Budget,
    DynamicBeamConfig,
    beam,
    bfs,
    cdfs,
    dfs,
    dynamic_beam,
    gbfs,
    heuristic_dfs,
    vcbfs,
    vcdfs,
    vcdfsa,
)
from heuristic.data import load_measurements
from heuristic.experiment import source_digest
from heuristic.model import FrozenPredictor
from nas_space import ArchitectureTree, architecture_string

# Human-editable output defaults. CLI --[no-] switches override these.
PRINT_RUNS = True
PRINT_SUMMARY = True
WRITE_JSON = True
WRITE_CSV = False


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    algorithm: str
    parameters: dict


DEFAULT_SPECS = (
    AlgorithmSpec("DFS", "dfs", {}),
    AlgorithmSpec("BFS", "bfs", {}),
    AlgorithmSpec("DFS-H", "heuristic_dfs", {}),
    AlgorithmSpec("GBFS", "gbfs", {}),
    AlgorithmSpec("CDFS-2", "cdfs", {"k": 2}),
    AlgorithmSpec("CDFS-4", "cdfs", {"k": 4}),
    AlgorithmSpec("Beam-4", "beam", {"width": 4}),
    AlgorithmSpec("Beam-16", "beam", {"width": 16}),
    AlgorithmSpec(
        "DynamicBeam-Entropy-4-16",
        "dynamic_beam",
        {
            "policy": "entropy",
            "minimum_width": 4,
            "maximum_width": 16,
            "slope": 4.0,
            "intercept": 0.0,
            "temperature": 1.0,
        },
    ),
    AlgorithmSpec(
        "DynamicBeam-StdDev-4-16",
        "dynamic_beam",
        {
            "policy": "standard_deviation",
            "minimum_width": 4,
            "maximum_width": 16,
            "slope": -100.0,
            "intercept": 16.0,
            "top_k": 5,
            "temperature": 1.0,
        },
    ),
    AlgorithmSpec("VCDFS-20-40pct", "vcdfs", {"pmin": 20, "pmax": 40}),
    AlgorithmSpec("VCDFS-20-80pct", "vcdfs", {"pmin": 20, "pmax": 80}),
    AlgorithmSpec("VCDFS-20-100pct", "vcdfs", {"pmin": 20, "pmax": 100}),
    AlgorithmSpec("VCDFS-A-1-2", "vcdfsa", {"kmin": 1, "kmax": 2}),
    AlgorithmSpec("VCDFS-A-1-3", "vcdfsa", {"kmin": 1, "kmax": 3}),
    AlgorithmSpec("VCDFS-A-2-5", "vcdfsa", {"kmin": 2, "kmax": 5}),
    AlgorithmSpec("VCBFS-20-40pct", "vcbfs", {"pmin": 20, "pmax": 40}),
    AlgorithmSpec("VCBFS-20-80pct", "vcbfs", {"pmin": 20, "pmax": 80}),
)

RUNNERS = {
    "dfs": dfs,
    "bfs": bfs,
    "heuristic_dfs": heuristic_dfs,
    "gbfs": gbfs,
    "cdfs": cdfs,
    "beam": beam,
    "vcdfs": vcdfs,
    "vcdfsa": vcdfsa,
    "vcbfs": vcbfs,
}


def save_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def load_specs(path=None):
    if path is None:
        return list(DEFAULT_SPECS)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Algorithm specification must be a nonempty JSON list")
    specs = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"name", "algorithm", "parameters"}:
            raise ValueError("Each specification needs name, algorithm, and parameters")
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError("Algorithm name must be nonempty")
        if not isinstance(row["parameters"], dict):
            raise ValueError("Algorithm parameters must be an object")
        if row["algorithm"] not in RUNNERS and row["algorithm"] != "dynamic_beam":
            raise ValueError(f"Unknown algorithm: {row['algorithm']}")
        specs.append(AlgorithmSpec(row["name"], row["algorithm"], row["parameters"]))
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Algorithm names must be unique")
    return specs


def execute(spec, problem, budget, event=None):
    if spec.algorithm == "dynamic_beam":
        parameters = dict(spec.parameters)
        config = DynamicBeamConfig(**parameters)
        return dynamic_beam(problem, config, budget=budget, event=event)
    return RUNNERS[spec.algorithm](problem, budget=budget, event=event, **dict(spec.parameters))


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
        if kind in {"retain", "beam", "vcbfs", "dynamic_beam"}:
            self.widths[int(event["k"])] += 1
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


def current_rss_bytes():
    """Current process resident/working-set bytes on Windows or procfs systems."""
    if platform.system() == "Windows":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("quota_peak_paged", ctypes.c_size_t),
                ("quota_paged", ctypes.c_size_t),
                ("quota_peak_nonpaged", ctypes.c_size_t),
                ("quota_nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t),
                ("private_usage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.working_set)
    statm = Path("/proc/self/statm")
    if statm.exists():
        import os

        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return None


class RssSampler:
    def __init__(self, interval=0.002):
        self.interval = interval
        self.start = current_rss_bytes()
        self.peak = self.start
        self._stop = Event()
        self._thread = Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(self.interval):
            value = current_rss_bytes()
            if value is not None and (self.peak is None or value > self.peak):
                self.peak = value

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        value = current_rss_bytes()
        if value is not None and (self.peak is None or value > self.peak):
            self.peak = value


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


def checkpoint_metadata(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if not all(key in metadata for key in ("split_seed", "model_seed", "data_sha256")):
        raise ValueError("Checkpoint is missing split/model/data provenance")
    return metadata


def format_bytes(value):
    if value is None:
        return "unavailable"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def print_run(row):
    print(f"\nRESULT {row['algorithm']}  (split {row['split_seed']}, model {row['model_seed']})")
    print(
        f"  Quality    best={row['reported_best_error']:.4f}%  "
        f"regret={row['regret_pp']:.4f} pp  stop={row['stop']}"
    )
    print(
        f"  Search     queries={row['queries']}/{row['query_budget']}  "
        f"expanded={row['expanded']}  scored={row['scored']}  pruned={row['pruned']}  "
        f"pending-peak={row['peak_pending_entries']}"
    )
    print(
        f"  Resources  time={row['runtime_seconds']:.4f} s  "
        f"Python-peak={format_bytes(row['python_peak_bytes'])}  "
        f"RSS-delta={format_bytes(row['rss_peak_delta_bytes'])}",
        flush=True,
    )


def print_final_summary(rows):
    print("\nFINAL SPLIT-AWARE SUMMARY", flush=True)
    print(
        "Rank: mean regret >> mean RSS delta >> mean runtime. "
        "Expansions are shown but do not affect rank."
    )
    header = (
        f"{'#':>2}  {'Algorithm':<22} {'N':>2}  {'Regret':>9}  {'Time':>8}  "
        f"{'Expand':>7}  {'Score':>7}  {'Query':>5}  {'Pruned':>7}  "
        f"{'Pending':>7}  {'RSS Δ':>9}"
    )
    print(header)
    print("-" * len(header))
    ordered = sorted(
        rows,
        key=lambda value: (
            value["mean_regret_pp"],
            (
                value["mean_rss_peak_delta_bytes"]
                if value["mean_rss_peak_delta_bytes"] is not None
                else float("inf")
            ),
            value["mean_runtime_seconds"],
            value["algorithm"],
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        print(
            f"{rank:>2}  {row['algorithm']:<22} {row['independent_splits']:>2}  "
            f"{row['mean_regret_pp']:>7.4f}pp  {row['mean_runtime_seconds']:>7.4f}s  "
            f"{row['mean_expanded']:>7.1f}  {row['mean_scored']:>7.1f}  "
            f"{row['mean_queries']:>5.1f}  {row['mean_pruned']:>7.1f}  "
            f"{row['mean_peak_pending_entries']:>7.1f}  "
            f"{format_bytes(row['mean_rss_peak_delta_bytes']):>9}",
            flush=True,
        )


def run_checkpoint(
    directory, records, data_checksum, specs, query_budget, *, print_runs=PRINT_RUNS
):
    directory = Path(directory)
    metadata = checkpoint_metadata(directory / "model.json")
    if metadata["data_sha256"] != data_checksum:
        raise ValueError("Checkpoint and measurement data checksums differ")
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
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
        reported_best = (
            initial_incumbent
            if result.best_value is None
            else min(initial_incumbent, result.best_value)
        )
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
            "reported_best_error": reported_best,
            "oracle_minimum_error": oracle_minimum,
            "regret_pp": reported_best - oracle_minimum,
            "width_histogram": dict(sorted(events.widths.items())),
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


SCALAR_COLUMNS = (
    "checkpoint",
    "split_seed",
    "model_seed",
    "algorithm",
    "implementation",
    "query_budget",
    "stop",
    "runtime_seconds",
    "python_peak_bytes",
    "rss_peak_bytes",
    "rss_peak_delta_bytes",
    "peak_pending_entries",
    "expanded",
    "generated",
    "scored",
    "queries",
    "unique_terminal_states",
    "new_terminal_states",
    "rejected",
    "pruned",
    "max_depth",
    "search_best_error",
    "initial_known_error",
    "reported_best_error",
    "oracle_minimum_error",
    "regret_pp",
    "mean_confidence",
    "confidence_saturation_fraction",
    "mean_controller_statistic",
)


def save_csv(path, rows, columns):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def summarize(rows):
    per_split = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], row["split_seed"])].append(row)
    metrics = (
        "regret_pp",
        "runtime_seconds",
        "python_peak_bytes",
        "rss_peak_bytes",
        "rss_peak_delta_bytes",
        "peak_pending_entries",
        "expanded",
        "generated",
        "scored",
        "queries",
        "pruned",
    )
    for (algorithm, split_seed), members in sorted(groups.items()):
        aggregate = {"algorithm": algorithm, "split_seed": split_seed, "model_runs": len(members)}
        aggregate.update({key: statistics.fmean(row[key] for row in members) for key in metrics})
        per_split.append(aggregate)
    final = []
    by_algorithm = defaultdict(list)
    for row in per_split:
        by_algorithm[row["algorithm"]].append(row)
    for algorithm, members in sorted(by_algorithm.items()):
        aggregate = {
            "algorithm": algorithm,
            "independent_splits": len(members),
            "model_runs": sum(row["model_runs"] for row in members),
        }
        for key in metrics:
            values = [row[key] for row in members]
            aggregate[f"mean_{key}"] = statistics.fmean(values)
            aggregate[f"sd_across_split_means_{key}"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        final.append(aggregate)
    return per_split, final


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
        save_json(args.output / "manifest.json", manifest)
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
        save_json(args.output / "runs.json", rows)
        save_json(args.output / "split_summary.json", per_split)
        save_json(args.output / "summary.json", summary)
    if args.write_csv and per_split:
        save_csv(args.output / "runs.csv", rows, SCALAR_COLUMNS)
        save_csv(args.output / "split_summary.csv", per_split, tuple(per_split[0]))
    if args.write_csv and summary:
        save_csv(args.output / "summary.csv", summary, tuple(summary[0]))


if __name__ == "__main__":
    main()
