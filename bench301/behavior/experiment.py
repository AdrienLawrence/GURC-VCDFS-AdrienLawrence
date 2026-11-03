"""Paired, resumable NAS-Bench-301 search behavior studies."""

if __name__ == "__main__" and not __package__:
    raise SystemExit(
        "Open one of bench301/runs/analyze_bench301_*.py and press Run. "
        "This implementation module needs package imports and study settings."
    )

import argparse
import gc
import hashlib
import platform
import statistics
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import torch

from algs import Budget
from bench301.empirical import SearchObserver
from bench301.heuristic.data import load_measurements
from bench301.heuristic.model import FrozenPredictor, checkpoint_metadata
from bench301.oracle import TerminalOracle
from bench301.representation import ENCODING, genotype_json, state_to_config
from bench301.space import ArchitectureTree
from resource_measurement import RssSampler, current_rss_bytes
from schemas import load_payload, write_artifact
from search_registry import execute

from .analysis import anytime_mean_error, print_summary, save_csv, split_aware_summary
from .presets import parameter_specs

KIND = {
    "manifest": "nb301_behavior_manifest",
    "runs": "nb301_behavior_runs",
    "split_summary": "nb301_behavior_split_summary",
    "summary": "nb301_behavior_summary",
}


class ResourceGuard:
    """Cooperative limits checked at every visited state and frontier update.

    A single in-flight surrogate prediction can cross the time limit. Report the
    measured overshoot; never present a limited run as natural exhaustion.
    """

    def __init__(self, seconds, rss_delta_mib):
        self.start = time.perf_counter()
        self.seconds = seconds
        self.initial_rss = current_rss_bytes()
        self.memory_bytes = int(rss_delta_mib * 1024 * 1024)
        if self.memory_bytes and self.initial_rss is None:
            raise RuntimeError("RSS sampling is unavailable; cannot enforce memory guard")

    def __call__(self):
        if self.seconds is not None and time.perf_counter() - self.start >= self.seconds:
            return "time_limit"
        if self.initial_rss is not None and self.memory_bytes is not None:
            current = current_rss_bytes()
            if current is not None and current - self.initial_rss >= self.memory_bytes:
                return "memory_limit"
        return None


class TimedObjective:
    def __init__(self, oracle):
        self.oracle = oracle
        self.seconds = 0.0

    def __call__(self, state):
        start = time.perf_counter()
        try:
            return self.oracle(state)
        finally:
            self.seconds += time.perf_counter() - start


def select_checkpoints(study, models_per_split):
    candidates = sorted(path for path in Path(study).glob("split-*-model-*") if path.is_dir())
    groups = {}
    for directory in candidates:
        if not (directory / "model.json").is_file():
            raise ValueError(f"Incomplete checkpoint: {directory}")
        metadata = checkpoint_metadata(directory / "model.json")
        key = int(metadata["split_seed"])
        groups.setdefault(key, []).append((int(metadata["model_seed"]), directory, metadata))
    selected = []
    for split_seed in sorted(groups):
        models = sorted(groups[split_seed])
        selected.extend(models if models_per_split is None else models[:models_per_split])
    if not selected:
        raise ValueError(f"No complete Bench301 checkpoints in {study}")
    return selected


def run_one(spec, budget, predictor, oracle, limits, anytime_horizon):
    """Measure one algorithm. Every run starts with an empty terminal cache."""
    oracle.clear()
    objective = TimedObjective(oracle)
    observer = SearchObserver()
    gc.collect()
    tracemalloc.start()
    try:
        with RssSampler() as rss:
            guard = ResourceGuard(limits["seconds"], limits["rss_delta_mib"])
            started = time.perf_counter()
            observer.started = started
            result = execute(
                spec,
                ArchitectureTree(predictor, objective, batch_scores=True),
                Budget(
                    queries=None if budget == -1 else budget,
                    expansions=limits["expansions"],
                    stop_if=guard,
                ),
                observer,
            )
            elapsed = time.perf_counter() - started
        _, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    found = result.best_state is not None
    final = observer.improvements[-1] if observer.improvements else None
    if found != (final is not None):
        raise RuntimeError("Terminal events and best-state result disagree")
    if result.queries != observer.queries or result.scored != observer.scored:
        raise RuntimeError("Search event accounting disagrees with result counters")
    best_state = list(result.best_state) if found else None
    return {
        "algorithm": spec.name,
        "implementation": spec.algorithm,
        "parameters": spec.parameters,
        "query_budget": budget,
        "budget_mode": "guarded_exhaustive" if budget == -1 else "bounded",
        "stop": result.stop,
        "found": found,
        "runtime_seconds": elapsed,
        "oracle_seconds": objective.seconds,
        "python_peak_bytes": python_peak,
        "rss_peak_bytes": rss.peak,
        "rss_peak_delta_bytes": (None if rss.start is None else max(0, rss.peak - rss.start)),
        "memory_scope": "main process only; isolated surrogate worker excluded",
        "expanded": result.expanded,
        "generated": result.generated,
        "scored": result.scored,
        "queries": result.queries,
        "surrogate_calls": oracle.surrogate_calls,
        "rejected": result.rejected,
        "pruned": result.pruned,
        "max_depth": result.max_depth,
        "peak_pending_entries": result.peak_pending,
        "retained_fraction": (
            (result.generated - result.pruned) / result.generated if result.generated else 1.0
        ),
        "scores_per_query": result.scored / result.queries if result.queries else None,
        "expansions_per_query": result.expanded / result.queries if result.queries else None,
        "best_error": result.best_value,
        "best_state": best_state,
        "best_genotype": genotype_json(result.best_state) if found else None,
        "best_config": state_to_config(result.best_state) if found else None,
        "anytime_mean_error": anytime_mean_error(observer.improvements, anytime_horizon),
        "anytime_horizon": anytime_horizon,
        "best_so_far_curve": observer.improvements,
        "query_of_final_best": final["query"] if final else None,
        "expansions_to_final_best": final["expansions"] if final else None,
        "scores_to_final_best": final["scored"] if final else None,
        "seconds_to_final_best": final["seconds"] if final else None,
        "width_histogram": {str(key): value for key, value in sorted(observer.widths.items())},
        "mean_confidence": (
            statistics.fmean(observer.confidences) if observer.confidences else None
        ),
    }


def _run_key(row):
    return row["checkpoint"], row["algorithm"], int(row["query_budget"])


def open_or_resume(output, manifest, planned):
    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)
        write_artifact(output / "manifest.json", KIND["manifest"], manifest)
        return []
    if not output.is_dir() or not (output / "manifest.json").is_file():
        raise ValueError("Existing output is not a resumable Bench301 study")
    saved = load_payload(output / "manifest.json", KIND["manifest"])
    if saved != manifest:
        different = sorted(
            key for key in saved.keys() | manifest.keys() if saved.get(key) != manifest.get(key)
        )
        raise ValueError(f"Cannot resume a changed study plan: {different}")
    partial, complete = output / "runs.partial.json", output / "runs.json"
    source = partial if partial.exists() else complete
    rows = load_payload(source, KIND["runs"]) if source.exists() else []
    keys = [_run_key(row) for row in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= set(planned):
        raise ValueError("Saved run keys are duplicate or outside the study plan")
    if complete.exists() and not partial.exists() and set(keys) != set(planned):
        raise ValueError("Complete runs.json is missing planned rows")
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("bench301/artifacts/measurements.json"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("v-family", "baselines"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--budgets", type=int, nargs="+")
    mode.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--models-per-split", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--oracle-model", type=Path, required=True)
    parser.add_argument("--oracle-python", type=Path)
    parser.add_argument("--oracle-backend", choices=("official", "minimal-xgb"), default="official")
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--max-rss-delta-mib", type=float, default=512)
    parser.add_argument("--max-expansions", type=int, default=5_000)
    parser.add_argument("--anytime-horizon", type=int, default=8)
    args = parser.parse_args()
    if args.models_per_split < 1 or args.checkpoint_every < 1:
        parser.error("model and checkpoint counts must be positive")
    if args.max_seconds <= 0 or args.max_rss_delta_mib <= 0 or args.max_expansions < 1:
        parser.error("resource guards must be positive")
    if args.anytime_horizon < 1:
        parser.error("anytime horizon must be positive")
    if args.budgets and (
        any(value < 1 for value in args.budgets) or len(set(args.budgets)) != len(args.budgets)
    ):
        parser.error("query budgets must be unique positive integers")
    budgets = [-1] if args.exhaustive else sorted(args.budgets)
    checkpoints = select_checkpoints(args.study, args.models_per_split)
    _, data_sha256 = load_measurements(args.data)
    if any(metadata["data_sha256"] != data_sha256 for _, _, metadata in checkpoints):
        parser.error("A checkpoint was trained on a different measurement source")
    specs = parameter_specs(args.preset)
    limits = {
        "seconds": args.max_seconds,
        "rss_delta_mib": args.max_rss_delta_mib,
        "expansions": args.max_expansions,
    }
    oracle = TerminalOracle.load(args.oracle_model, args.oracle_python, backend=args.oracle_backend)
    try:
        manifest = {
            "scope": "NAS-BENCH-301 DARTS SEARCH BEHAVIOR",
            "benchmark": "nasbench301",
            "encoding": ENCODING,
            "data_sha256": data_sha256,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "preset": args.preset,
            "models_per_split": args.models_per_split,
            "budgets": budgets,
            "checkpoint_directories": [directory.name for _, directory, _ in checkpoints],
            "checkpoint_sha256": {
                directory.name: _sha256(directory / "model.json") for _, directory, _ in checkpoints
            },
            "specifications": [asdict(spec) for spec in specs],
            "oracle": oracle.provenance,
            "resource_limits": limits,
            "anytime_horizon": args.anytime_horizon,
            "metrics_version": 1,
            "search_scope": "full implicit DARTS space; objective is deterministic NB301 surrogate",
            "quality_scope": "observed surrogate validation error; no known global optimum",
            "memory_scope": "main Python process, including instrumentation; oracle worker excluded",
            "runtime_scope": "instrumented online search including terminal oracle calls",
            "batch_scoring": True,
            "batch_score_precision": "float32 predictions may differ from scalar inference near ties",
            "guard_semantics": "cooperative between states; one oracle call may overshoot time/memory",
            "offline_training_excluded": True,
        }
        planned = [
            (directory.name, spec.name, budget)
            for _, directory, _ in checkpoints
            for spec in specs
            for budget in budgets
        ]
        rows = open_or_resume(args.output, manifest, planned)
        completed = {_run_key(row) for row in rows}
        if len(completed) == len(planned) and (args.output / "runs.json").exists():
            print(f"NB301 BEHAVIOR | already complete: {len(rows)} runs in {args.output}")
            return
        print(
            f"NB301 BEHAVIOR | checkpoints={len(checkpoints)} configs={len(specs)} "
            f"budgets={len(budgets)} runs={len(planned)}",
            flush=True,
        )
        print(f"RESUME | completed={len(rows)} remaining={len(planned) - len(rows)}", flush=True)
        for _, directory, metadata in checkpoints:
            if all(
                (directory.name, spec.name, budget) in completed
                for spec in specs
                for budget in budgets
            ):
                continue
            predictor = FrozenPredictor.load(directory / "model.json")
            predictor(())  # Load and warm the frozen model before search timing.
            for spec in specs:
                for budget in budgets:
                    key = (directory.name, spec.name, budget)
                    if key in completed:
                        continue
                    row = run_one(
                        spec,
                        budget,
                        predictor,
                        oracle,
                        limits,
                        args.anytime_horizon if budget == -1 else budget,
                    )
                    row.update(
                        {
                            "checkpoint": directory.name,
                            "split_seed": int(metadata["split_seed"]),
                            "model_seed": int(metadata["model_seed"]),
                        }
                    )
                    rows.append(row)
                    completed.add(key)
                    quality = f"{row['best_error']:.4f}pp" if row["found"] else "no terminal"
                    print(
                        f"[{len(rows)}/{len(planned)}] {directory.name} {spec.name} "
                        f"B={'ALL' if budget == -1 else budget} best={quality} "
                        f"q={row['queries']} expanded={row['expanded']} scored={row['scored']} "
                        f"stop={row['stop']} time={row['runtime_seconds']:.2f}s",
                        flush=True,
                    )
                    if len(rows) % args.checkpoint_every == 0:
                        write_artifact(
                            args.output / "runs.partial.json", KIND["runs"], rows, replace=True
                        )
        if len(rows) != len(planned):
            raise RuntimeError("Incomplete Bench301 study plan")
        partial = args.output / "runs.partial.json"
        write_artifact(partial, KIND["runs"], rows, replace=True)
        partial.replace(args.output / "runs.json")
        per_split, summary = split_aware_summary(rows)
        write_artifact(
            args.output / "split_summary.json",
            KIND["split_summary"],
            per_split,
            replace=True,
        )
        write_artifact(args.output / "summary.json", KIND["summary"], summary, replace=True)
        save_csv(args.output / "runs.csv", rows)
        save_csv(args.output / "split_summary.csv", per_split)
        save_csv(args.output / "summary.csv", summary)
        print_summary(summary)
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
