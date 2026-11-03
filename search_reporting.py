"""Search metrics, console tables, CSV output, and split-aware aggregation."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def format_bytes(value):
    if value is None:
        return "unavailable"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def quality_metrics(search_best, initial_incumbent, oracle_minimum):
    """Primary search-only quality plus a separately labeled incumbent diagnostic."""
    if search_best is None:
        raise RuntimeError("Search produced no terminal solution; regret is undefined")
    augmented = min(initial_incumbent, search_best)
    return {
        "reported_best_error": search_best,
        "regret_pp": search_best - oracle_minimum,
        "incumbent_augmented_best_error": augmented,
        "incumbent_augmented_regret_pp": augmented - oracle_minimum,
    }


def json_histogram(histogram):
    return {str(bucket): count for bucket, count in sorted(histogram.items())}


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
    print("Rank: mean regret >> mean RSS delta >> mean runtime. Expansions do not affect rank.")
    header = (
        f"{'#':>2}  {'Algorithm':<22} {'N':>2}  {'Regret':>9}  {'Time':>8}  "
        f"{'Expand':>7}  {'Score':>7}  {'Query':>5}  {'Pruned':>7}  "
        f"{'Pending':>7}  {'RSS Δ':>9}"
    )
    print(header)
    print("-" * len(header))
    ordered = sorted(
        rows,
        key=lambda row: (
            row["mean_regret_pp"],
            row["mean_rss_peak_delta_bytes"]
            if row["mean_rss_peak_delta_bytes"] is not None
            else float("inf"),
            row["mean_runtime_seconds"],
            row["algorithm"],
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
    "search_best_architecture",
    "initial_known_error",
    "reported_best_error",
    "oracle_minimum_error",
    "regret_pp",
    "incumbent_augmented_best_error",
    "incumbent_augmented_regret_pp",
    "mean_confidence",
    "confidence_saturation_fraction",
    "mean_controller_statistic",
)


def save_csv(path, rows, columns):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    """Average repeated model fits inside splits before comparing independent splits."""
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
    groups = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], row["split_seed"])].append(row)
    per_split = []
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
