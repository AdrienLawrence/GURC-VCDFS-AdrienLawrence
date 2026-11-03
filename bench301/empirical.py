"""Online search with frozen NB301 models; no global-optimum claims."""

import argparse
import gc
import statistics
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from algs import Budget
from resource_measurement import RssSampler
from schemas import load_payload, write_artifact
from search_registry import execute
from search_reporting import format_bytes, json_histogram

from .heuristic.model import FrozenPredictor, checkpoint_metadata
from .oracle import TerminalOracle
from .representation import ENCODING, SEQUENCE_LENGTH, genotype_json, state_to_config
from .space import ArchitectureTree
from .specs import load_specs


class SearchObserver:
    def __init__(self):
        self.expanded = self.scored = self.queries = 0
        self.widths = Counter()
        self.confidences = []
        self.improvements = []
        self.started = time.perf_counter()

    def __call__(self, event):
        kind = event["kind"]
        if kind == "visit" and len(event["state"]) < SEQUENCE_LENGTH:
            self.expanded += 1
        elif kind == "rank":
            self.scored += event["count"]
        elif kind == "terminal":
            self.queries += 1
            value = float(event["value"])
            if not self.improvements or value < self.improvements[-1]["value"]:
                self.improvements.append(
                    {
                        "query": self.queries,
                        "expansions": self.expanded,
                        "scored": self.scored,
                        "seconds": time.perf_counter() - self.started,
                        "value": value,
                        "state": list(event["state"]),
                    }
                )
        if kind in {"retain", "beam", "dynamic_beam", "vcbfs_beam"}:
            self.widths[event["k"]] += 1
        if kind in {"vcbfs", "vcbfsn"}:
            self.widths.update(group["k"] for group in event["groups"])
        values = [event.get("confidence")]
        if kind == "vcbfsn":
            values = [group.get("confidence") for group in event["groups"]]
        self.confidences.extend(float(value) for value in values if value is not None)


def run_one(spec, predictor, oracle, query_budget, reference=None):
    budget = Budget(queries=query_budget)
    oracle.clear()
    observer = SearchObserver()
    started = time.perf_counter()
    result = execute(spec, ArchitectureTree(predictor, oracle), budget, observer)
    elapsed = time.perf_counter() - started
    oracle_calls = oracle.surrogate_calls
    # Fresh caches in both runs prevent sharing terminal outcomes between algorithms.
    oracle.clear()
    gc.collect()
    tracemalloc.start()
    try:
        with RssSampler() as rss:
            replay = execute(spec, ArchitectureTree(predictor, oracle), budget)
        _, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if asdict(result) != asdict(replay):
        raise RuntimeError(f"Nondeterministic replay for {spec.name}")
    if result.best_state is None or not observer.improvements:
        raise RuntimeError(f"{spec.name} produced no complete architecture")
    final = observer.improvements[-1]
    row = {
        "algorithm": spec.name,
        "implementation": spec.algorithm,
        "parameters": spec.parameters,
        "query_budget": query_budget,
        "stop": result.stop,
        "runtime_seconds": elapsed,
        "python_peak_bytes": python_peak,
        "rss_peak_bytes": rss.peak,
        "rss_peak_delta_bytes": None if rss.start is None else max(0, rss.peak - rss.start),
        "memory_scope": "main process only; isolated oracle worker excluded",
        "peak_pending_entries": result.peak_pending,
        "expanded": result.expanded,
        "generated": result.generated,
        "scored": result.scored,
        "queries": result.queries,
        "rejected": result.rejected,
        "pruned": result.pruned,
        "max_depth": result.max_depth,
        "search_best_error": result.best_value,
        "best_state": list(result.best_state),
        "best_genotype": genotype_json(result.best_state),
        "best_config": state_to_config(result.best_state),
        "surrogate_calls": oracle_calls,
        "width_histogram": json_histogram(observer.widths),
        "mean_confidence": statistics.fmean(observer.confidences) if observer.confidences else None,
        "confidence_saturation_fraction": (
            sum(value in {0, 1} for value in observer.confidences) / len(observer.confidences)
            if observer.confidences
            else None
        ),
        "best_so_far_curve": observer.improvements,
        "query_of_final_best": final["query"],
        "expansions_to_final_best": final["expansions"],
        "scores_to_final_best": final["scored"],
        "seconds_to_final_best": final["seconds"],
    }
    if reference is not None:
        values = [item["error"] for item in reference["records"]]
        if not values:
            raise ValueError("Reference sample is empty")
        row["reference_gap_pp"] = result.best_value - min(values)
        row["reference_percentile"] = (
            100 * sum(value <= result.best_value for value in values) / len(values)
        )
    return row


def summarize(rows):
    metrics = (
        "search_best_error",
        "runtime_seconds",
        "python_peak_bytes",
        "rss_peak_delta_bytes",
        "expanded",
        "scored",
        "queries",
        "pruned",
        "peak_pending_entries",
    )
    by_split = defaultdict(list)
    for row in rows:
        by_split[(row["algorithm"], row["split_seed"])].append(row)
    per_algorithm = defaultdict(list)
    for (algorithm, split_seed), members in sorted(by_split.items()):
        item = {"algorithm": algorithm, "split_seed": split_seed, "model_runs": len(members)}
        for metric in metrics:
            values = [row[metric] for row in members if row[metric] is not None]
            item[metric] = statistics.fmean(values) if values else None
        per_algorithm[algorithm].append(item)
    summary = []
    for algorithm, splits in sorted(per_algorithm.items()):
        item = {
            "algorithm": algorithm,
            "independent_splits": len(splits),
            "model_runs": sum(row["model_runs"] for row in splits),
        }
        for metric in metrics:
            values = [row[metric] for row in splits if row[metric] is not None]
            item["mean_" + metric] = statistics.fmean(values) if values else None
            item["sd_across_split_means_" + metric] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", type=Path)
    selection.add_argument("--study", type=Path)
    parser.add_argument("--oracle-model", type=Path, required=True)
    parser.add_argument("--oracle-python", type=Path)
    parser.add_argument("--oracle-backend", choices=("official", "minimal-xgb"), default="official")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-json", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.queries < 1:
        parser.error("--queries must be positive")
    if args.write_json and args.output is None:
        parser.error("--output required for --write-json")
    checkpoints = (
        [args.checkpoint] if args.checkpoint else sorted(args.study.glob("split-*-model-*"))
    )
    if not checkpoints or any(
        not (directory / "model.json").is_file() for directory in checkpoints
    ):
        parser.error("No valid NB301 model.json checkpoint directories found")
    if args.output and args.output.exists():
        parser.error("Output directory already exists; choose a new output")
    metadata = [checkpoint_metadata(directory / "model.json") for directory in checkpoints]
    if len({item["data_sha256"] for item in metadata}) != 1:
        parser.error("Study checkpoints have different measurement sources")
    specs = load_specs(args.specs)
    oracle = TerminalOracle.load(args.oracle_model, args.oracle_python, backend=args.oracle_backend)
    try:
        reference = load_payload(args.reference, "nb301_reference") if args.reference else None
        if reference and reference["oracle"] != oracle.provenance:
            raise ValueError("Reference and search must use the same deterministic oracle/runtime")
        manifest = {
            "benchmark": "nasbench301",
            "encoding": ENCODING,
            "oracle": oracle.provenance,
            "checkpoint_directories": [str(path) for path in checkpoints],
            "checkpoints": metadata,
            "query_budget": args.queries,
            "specifications": [asdict(spec) for spec in specs],
            "offline_cost": "FP acquisition/training excluded from online terminal query budget",
            "offline_train_architectures": [item["train_architectures"] for item in metadata],
            "quality_scope": "observed surrogate objective; no known global optimum",
            "retention_rule": "shared unchanged absolute interpolation then local clip",
            "configuration_scope": "initial provisional controls, not tuned settings",
        }
        rows = []
        for directory, item in zip(checkpoints, metadata, strict=True):
            predictor = FrozenPredictor.load(directory / "model.json")
            predictor(())  # Prefix-only warmup; no objective call.
            for spec in specs:
                print(f"Running NB301 {directory.name}: {spec.name}", flush=True)
                row = run_one(spec, predictor, oracle, args.queries, reference)
                row.update(
                    {
                        "checkpoint": directory.name,
                        "split_seed": item["split_seed"],
                        "model_seed": item["model_seed"],
                    }
                )
                rows.append(row)
                print(
                    f"{spec.name}: best={row['search_best_error']:.4f}pp queries={row['queries']}/{args.queries} "
                    f"expanded={row['expanded']} scored={row['scored']} stop={row['stop']} "
                    f"time={row['runtime_seconds']:.3f}s Python={format_bytes(row['python_peak_bytes'])}",
                    flush=True,
                )
        summary = summarize(rows)
        print(
            "NB301 SUMMARY | mean observed surrogate error; global optimum is unknown", flush=True
        )
        for row in sorted(
            summary, key=lambda item: (item["mean_search_best_error"], item["mean_runtime_seconds"])
        ):
            print(
                f"{row['algorithm']:<24} best={row['mean_search_best_error']:.4f}pp "
                f"time={row['mean_runtime_seconds']:.3f}s expanded={row['mean_expanded']:.1f}",
                flush=True,
            )
        if args.write_json:
            args.output.mkdir(parents=True, exist_ok=False)
            write_artifact(args.output / "manifest.json", "nb301_search_manifest", manifest)
            write_artifact(args.output / "runs.json", "nb301_search_runs", rows)
            write_artifact(args.output / "summary.json", "nb301_search_summary", summary)
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
