"""Sweep V-family retention and query budgets on held-out NAS-Bench-201."""

import argparse
import csv
import gc
import platform
import statistics
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from algs import Budget
from bench201.behavior.analysis import BehaviorObserver
from bench201.heuristic.data import load_measurements
from bench201.heuristic.model import FrozenPredictor, checkpoint_metadata
from bench201.space import ArchitectureTree, architecture_string
from bench201.specs import DEFAULT_SPECS, execute
from resource_measurement import RssSampler
from schemas import load_payload, write_artifact


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    algorithm: str
    parameters: dict


DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 100)
PLAN_FIELDS = (
    "data_sha256",
    "preset",
    "models_per_split",
    "budgets",
    "checkpoint_directories",
    "specifications",
    "metrics_version",
)


def parameter_specs(preset="full"):
    valid = {"quick", "full", "absolute-ablation", "baselines"}
    if preset not in valid:
        raise ValueError(f"preset must be one of {sorted(valid)}")
    if preset == "baselines":
        excluded = {"vcdfs", "vcdfsa", "vcbfs", "vcbfsa", "vcbfsn", "vcbfsna", "vcbfs_beam"}
        return [
            BehaviorSpec(spec.name, spec.algorithm, spec.parameters)
            for spec in DEFAULT_SPECS
            if spec.algorithm not in excluded
        ]
    proportional = (
        [(20, 40), (20, 80), (20, 100)]
        if preset == "quick"
        else [(lower, upper) for lower in range(20, 101, 20) for upper in range(lower, 101, 20)]
    )
    absolute = (
        [(1, 2), (1, 3), (2, 5)]
        if preset == "quick"
        else [(lower, upper) for lower in range(1, 6) for upper in range(lower, 6)]
    )
    specs = []
    if preset != "absolute-ablation":
        for algorithm in ("vcdfs", "vcbfs", "vcbfsn"):
            label = algorithm.upper()
            specs.extend(
                BehaviorSpec(
                    f"{label}-{lower}-{upper}pct",
                    algorithm,
                    {"pmin": lower, "pmax": upper},
                )
                for lower, upper in proportional
            )
    for algorithm in ("vcdfsa", "vcbfsa", "vcbfsna"):
        label = algorithm.upper().replace("A", "-A")
        specs.extend(
            BehaviorSpec(
                f"{label}-{lower}-{upper}",
                algorithm,
                {"kmin": lower, "kmax": upper},
            )
            for lower, upper in absolute
        )
    return specs


def select_checkpoints(checkpoints, models_per_split=None):
    """Keep the lowest model seeds per split; ``None`` preserves every checkpoint."""
    checkpoints = list(checkpoints)
    if models_per_split is None:
        return checkpoints
    if type(models_per_split) is not int or models_per_split < 1:
        raise ValueError("models_per_split must be a positive integer or None")
    groups = defaultdict(list)
    for checkpoint in checkpoints:
        metadata = checkpoint_metadata(checkpoint / "model.json")
        groups[int(metadata["split_seed"])].append((int(metadata["model_seed"]), checkpoint))
    return [
        checkpoint
        for split_seed in sorted(groups)
        for _, checkpoint in sorted(groups[split_seed])[:models_per_split]
    ]


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


def run_one(spec, budget, predictor, objectives, allowed, optimal_states, oracle):
    def problem():
        return ArchitectureTree(predictor, objectives.__getitem__, allowed=tuple(allowed))

    observer = BehaviorObserver(optimal_states, oracle)
    started = time.perf_counter()
    result = execute(spec, problem(), Budget(queries=budget), observer)
    elapsed = time.perf_counter() - started
    if result.best_value is None:
        label = "exhaustion" if budget is None else f"query budget {budget}"
        raise RuntimeError(f"{spec.name} produced no terminal at {label}")

    # Measure memory on a deterministic replay so tracemalloc and RSS sampling
    # do not alter the existing clean runtime measurement or observer timing.
    gc.collect()
    tracemalloc.start()
    with RssSampler() as rss:
        memory_result = execute(spec, problem(), Budget(queries=budget))
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not _same_result(result, memory_result):
        raise RuntimeError(f"Nondeterministic memory replay for {spec.name}")
    rss_delta = None if rss.start is None else max(0, rss.peak - rss.start)
    horizon = len(allowed) if budget is None else budget

    return {
        "algorithm": spec.name,
        "implementation": spec.algorithm,
        "parameters": spec.parameters,
        "query_budget": -1 if budget is None else budget,
        "budget_mode": "exhaustive" if budget is None else "bounded",
        "stop": result.stop,
        "runtime_seconds": elapsed,
        "python_peak_bytes": python_peak_bytes,
        "rss_peak_bytes": rss.peak,
        "rss_peak_delta_bytes": rss_delta,
        "expanded": result.expanded,
        "generated": result.generated,
        "scored": result.scored,
        "queries": result.queries,
        "pruned": result.pruned,
        "peak_pending_entries": result.peak_pending,
        "best_error": result.best_value,
        "best_architecture": architecture_string(result.best_state),
        "oracle_error": oracle,
        "final_regret_pp": result.best_value - oracle,
        "retained_fraction": (
            (result.generated - result.pruned) / result.generated if result.generated else 1.0
        ),
        "scores_per_query": result.scored / result.queries,
        "expansions_per_query": result.expanded / result.queries,
        **observer.metrics(horizon),
    }


SUMMARY_METRICS = (
    "final_regret_pp",
    "anytime_mean_regret_pp",
    "optimum_found",
    "any_optimum_pruned",
    "all_optima_pruned",
    "optimal_survival_fraction",
    "query_of_final_best",
    "expansions_to_final_best",
    "scores_to_final_best",
    "seconds_to_final_best",
    "runtime_seconds",
    "python_peak_bytes",
    "rss_peak_bytes",
    "rss_peak_delta_bytes",
    "expanded",
    "generated",
    "scored",
    "queries",
    "pruned",
    "peak_pending_entries",
    "retained_fraction",
    "scores_per_query",
    "expansions_per_query",
)


def split_aware_summary(rows):
    split_groups = defaultdict(list)
    for row in rows:
        split_groups[(row["algorithm"], row["query_budget"], row["split_seed"])].append(row)
    per_split = []
    for (algorithm, budget, split_seed), members in sorted(split_groups.items()):
        item = {
            "algorithm": algorithm,
            "implementation": members[0]["implementation"],
            "parameters": members[0]["parameters"],
            "query_budget": budget,
            "split_seed": split_seed,
            "model_runs": len(members),
        }
        item.update({key: statistics.fmean(row[key] for row in members) for key in SUMMARY_METRICS})
        per_split.append(item)
    final_groups = defaultdict(list)
    for row in per_split:
        final_groups[(row["algorithm"], row["query_budget"])].append(row)
    summary = []
    for (algorithm, budget), members in sorted(final_groups.items()):
        item = {
            "algorithm": algorithm,
            "implementation": members[0]["implementation"],
            "parameters": members[0]["parameters"],
            "query_budget": budget,
            "independent_splits": len(members),
            "model_runs": sum(row["model_runs"] for row in members),
        }
        for key in SUMMARY_METRICS:
            values = [row[key] for row in members]
            item[f"mean_{key}"] = statistics.fmean(values)
            item[f"sd_across_split_means_{key}"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(item)
    return per_split, summary


def print_row(row):
    optimum = "yes" if row["optimum_found"] else "no"
    budget = "ALL" if row["query_budget"] == -1 else str(row["query_budget"])
    print(
        f"{row['checkpoint']:<22} {row['algorithm']:<22} B={budget:>3} "
        f"regret={row['final_regret_pp']:.4f}pp anytime={row['anytime_mean_regret_pp']:.4f}pp "
        f"optimum={optimum:<3} "
        f"survive={row['optimal_paths_surviving']}/{row['optimal_architecture_count']} "
        f"q-best={row['query_of_final_best']:>3} scored={row['scored']:>5} "
        f"time={row['runtime_seconds']:.3f}s "
        f"rss+={row['rss_peak_delta_bytes'] / 1024:.1f}KiB",
        flush=True,
    )


def print_summary(summary, limit=5):
    print("\nBEHAVIOR SUMMARY | ranked by anytime regret, final regret, then scored nodes")
    budgets = sorted({row["query_budget"] for row in summary})
    for budget in budgets:
        rows = sorted(
            (row for row in summary if row["query_budget"] == budget),
            key=lambda row: (
                row["mean_anytime_mean_regret_pp"],
                row["mean_final_regret_pp"],
                row["mean_scored"],
            ),
        )
        label = "Exhaustive" if budget == -1 else f"Budget {budget}"
        print(f"\n{label} - top {min(limit, len(rows))}")
        print(
            "#  Algorithm                 Anytime    Final  Opt-found  All-pruned   Scored  Q-best  RSS+KiB"
        )
        for rank, row in enumerate(rows[:limit], start=1):
            print(
                f"{rank:<2} {row['algorithm']:<25} "
                f"{row['mean_anytime_mean_regret_pp']:>7.4f}  "
                f"{row['mean_final_regret_pp']:>7.4f}  "
                f"{row['mean_optimum_found']:>8.1%}  "
                f"{row['mean_all_optima_pruned']:>10.1%}  "
                f"{row['mean_scored']:>7.1f}  "
                f"{row['mean_query_of_final_best']:>6.1f}  "
                f"{row['mean_rss_peak_delta_bytes'] / 1024:>7.1f}"
            )


def save_csv(path, rows):
    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    if not scalar_rows:
        return
    # Extended artifacts may combine legacy rows with newer optional fields.
    # Preserve first-seen column order while accepting the union of all fields.
    fieldnames = list(dict.fromkeys(key for row in scalar_rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def run_key(row):
    """Identify one planned checkpoint/configuration/budget execution."""
    return row["checkpoint"], row["algorithm"], int(row["query_budget"])


def open_or_resume(output, manifest, planned_keys, *, extend=False):
    """Create a study or load compatible completed/partial rows."""
    output = Path(output)
    manifest_path = output / "manifest.json"
    partial_path = output / "runs.partial.json"
    complete_path = output / "runs.json"

    if not output.exists():
        output.mkdir(parents=True)
        write_artifact(manifest_path, "behavior_manifest", manifest)
        return []
    if not output.is_dir() or not manifest_path.exists():
        raise ValueError("Existing output is not a resumable behavior-study directory")

    # During an extension, partial contains new rows while runs.json still
    # represents the previously completed plan. Prefer the checkpoint.
    source = partial_path if partial_path.exists() else complete_path
    if not source.exists():
        rows = []
    else:
        rows = load_payload(source, "behavior_runs")
    saved_manifest = load_payload(manifest_path, "behavior_manifest")
    # Artifacts produced immediately before resume support had the same memory
    # fields but no explicit metrics_version. Recognize only that exact shape.
    legacy_memory_v2 = not rows or all("rss_peak_delta_bytes" in row for row in rows)
    differences = []
    for field in PLAN_FIELDS:
        if extend and field in {"preset", "specifications"}:
            continue
        saved_value = saved_manifest.get(field)
        if field == "metrics_version" and saved_value is None and legacy_memory_v2:
            saved_value = 2
        if saved_value != manifest[field]:
            differences.append(field)
    if differences:
        raise ValueError(
            "Cannot resume with a different experiment plan; changed fields: "
            + ", ".join(differences)
        )

    keys = [run_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Resume artifact contains duplicate runs")
    if extend:
        saved_specs = saved_manifest["specifications"]
        new_specs = manifest["specifications"]
        saved_by_name = {spec["name"]: spec for spec in saved_specs}
        conflicts = [
            spec["name"]
            for spec in new_specs
            if spec["name"] in saved_by_name and spec != saved_by_name[spec["name"]]
        ]
        if conflicts:
            raise ValueError(
                "Extension names conflict with different specifications: "
                + ", ".join(sorted(conflicts))
            )
        missing_specs = [spec for spec in new_specs if spec["name"] not in saved_by_name]
        if missing_specs:
            manifest = dict(manifest)
            manifest["preset"] = (
                saved_manifest["preset"]
                if saved_manifest["preset"] == manifest["preset"]
                else f"{saved_manifest['preset']}+{manifest['preset']}"
            )
            manifest["specifications"] = saved_specs + missing_specs
            write_artifact(manifest_path, "behavior_manifest", manifest, replace=True)
        planned_keys = list(keys) + list(planned_keys)
    unexpected = set(keys) - set(planned_keys)
    if unexpected:
        raise ValueError(f"Resume artifact contains {len(unexpected)} unplanned runs")
    if complete_path.exists() and not extend and set(keys) != set(planned_keys):
        raise ValueError("Completed runs.json does not contain the complete experiment plan")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/measurements.json"))
    parser.add_argument("--study", type=Path, default=Path("artifacts/full-study"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=(
            "quick",
            "full",
            "absolute-ablation",
            "baselines",
        ),
        default="quick",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run without expansion or terminal-query limits",
    )
    parser.add_argument("--models-per-split", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--extend-existing",
        action="store_true",
        help="Append a disjoint preset to a compatible behavior artifact",
    )
    args = parser.parse_args()
    if any(budget < 1 for budget in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        parser.error("--budgets must be unique positive integers")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    checkpoints = sorted(path for path in args.study.glob("split-*-model-*") if path.is_dir())
    if not checkpoints:
        parser.error("--study contains no checkpoints")
    checkpoints = select_checkpoints(checkpoints, args.models_per_split)
    specs = parameter_specs(args.preset)
    budget_values = [None] if args.exhaustive else sorted(args.budgets)
    manifest_budgets = ["exhaustive"] if args.exhaustive else budget_values
    records, checksum = load_measurements(args.data)
    objectives = {tuple(row["state"]): float(row["error"]) for row in records}
    manifest = {
        "scope": "HELD-OUT NAS-BENCH-201 SEARCH BEHAVIORAL SWEEP",
        "data_sha256": checksum,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "preset": args.preset,
        "models_per_split": args.models_per_split,
        "budgets": manifest_budgets,
        "checkpoint_directories": [path.name for path in checkpoints],
        "specifications": [asdict(spec) for spec in specs],
        "metrics_version": 2,
        "oracle_use": "post-run metrics only; held-out labels never enter search or fitting",
        "search_scope": "checkpoint test partition only",
        "independence": "model seeds are repeated fits; inference uses split means",
    }
    total = len(checkpoints) * len(specs) * len(budget_values)
    planned_keys = [
        (checkpoint.name, spec.name, -1 if budget is None else budget)
        for checkpoint in checkpoints
        for spec in specs
        for budget in budget_values
    ]
    rows = open_or_resume(args.output, manifest, planned_keys, extend=args.extend_existing)
    total = len({run_key(row) for row in rows} | set(planned_keys))
    completed = {run_key(row) for row in rows}
    print(
        f"SEARCH BEHAVIOR | checkpoints={len(checkpoints)} configs={len(specs)} "
        f"budgets={len(budget_values)} mode={'exhaustive' if args.exhaustive else 'bounded'} "
        f"runs={total}",
        flush=True,
    )
    if rows:
        print(f"RESUME | completed={len(rows)} remaining={total - len(rows)}", flush=True)
    for checkpoint in checkpoints:
        checkpoint_keys = {
            (checkpoint.name, spec.name, budget)
            for spec in specs
            for budget in (-1 if value is None else value for value in budget_values)
        }
        if checkpoint_keys <= completed:
            continue
        metadata = checkpoint_metadata(checkpoint / "model.json")
        if metadata["data_sha256"] != checksum:
            raise ValueError("Checkpoint/data checksum mismatch")
        split = load_payload(checkpoint / "split.json", "data_split", allow_legacy=True)
        allowed = [tuple(records[index]["state"]) for index in split["test"]]
        oracle = min(objectives[state] for state in allowed)
        optimal_states = {state for state in allowed if objectives[state] == oracle}
        predictor = FrozenPredictor.load(checkpoint / "model.json")
        predictor(())
        for spec in specs:
            for budget in budget_values:
                key = (checkpoint.name, spec.name, -1 if budget is None else budget)
                if key in completed:
                    continue
                row = run_one(spec, budget, predictor, objectives, allowed, optimal_states, oracle)
                row.update(
                    {
                        "checkpoint": checkpoint.name,
                        "split_seed": int(metadata["split_seed"]),
                        "model_seed": int(metadata["model_seed"]),
                    }
                )
                rows.append(row)
                completed.add(key)
                print(f"[{len(rows)}/{total}]", end=" ", flush=True)
                print_row(row)
                if len(rows) % args.checkpoint_every == 0:
                    write_artifact(
                        args.output / "runs.partial.json", "behavior_runs", rows, replace=True
                    )
    if len(rows) != total:
        raise RuntimeError(f"Behavior study produced {len(rows)} of {total} planned runs")
    partial_path = args.output / "runs.partial.json"
    complete_path = args.output / "runs.json"
    write_artifact(partial_path, "behavior_runs", rows, replace=True)
    partial_path.replace(complete_path)
    per_split, summary = split_aware_summary(rows)
    write_artifact(
        args.output / "split_summary.json", "behavior_split_summary", per_split, replace=True
    )
    write_artifact(args.output / "summary.json", "behavior_summary", summary, replace=True)
    save_csv(args.output / "runs.csv", rows)
    save_csv(args.output / "summary.csv", summary)
    print_summary(summary)
    print(f"Completed {len(rows)} runs -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
