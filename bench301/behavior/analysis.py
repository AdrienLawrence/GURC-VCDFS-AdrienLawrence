"""Quality and split-aware summaries without an assumed global optimum."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "best_error",
    "anytime_mean_error",
    "runtime_seconds",
    "oracle_seconds",
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
    "query_of_final_best",
    "expansions_to_final_best",
    "scores_to_final_best",
    "seconds_to_final_best",
)


def anytime_mean_error(improvements, horizon):
    """Carry forward the incumbent to a fixed query horizon after early exhaustion."""
    if not improvements:
        return None
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Anytime horizon must be positive")
    current = None
    position = 0
    total = 0.0
    for query in range(1, horizon + 1):
        while position < len(improvements) and improvements[position]["query"] <= query:
            current = improvements[position]["value"]
            position += 1
        if current is None:
            raise ValueError("The first terminal query is missing from the trace")
        total += current
    return total / horizon


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def split_aware_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], row["query_budget"], row["split_seed"])].append(row)
    per_split = []
    for (algorithm, budget, split_seed), members in sorted(groups.items()):
        item = {
            "algorithm": algorithm,
            "implementation": members[0]["implementation"],
            "parameters": members[0]["parameters"],
            "query_budget": budget,
            "split_seed": split_seed,
            "model_runs": len(members),
            "valid_runs": sum(row["found"] for row in members),
            "exhausted_runs": sum(row["stop"] == "exhausted" for row in members),
        }
        for key in METRICS:
            item[key] = _mean(row[key] for row in members)
        per_split.append(item)

    grouped = defaultdict(list)
    for row in per_split:
        grouped[(row["algorithm"], row["query_budget"])].append(row)
    summary = []
    for (algorithm, budget), splits in sorted(grouped.items()):
        item = {
            "algorithm": algorithm,
            "implementation": splits[0]["implementation"],
            "parameters": splits[0]["parameters"],
            "query_budget": budget,
            "independent_splits": len(splits),
            "model_runs": sum(row["model_runs"] for row in splits),
            "valid_runs": sum(row["valid_runs"] for row in splits),
            "exhausted_runs": sum(row["exhausted_runs"] for row in splits),
        }
        for key in METRICS:
            values = [row[key] for row in splits if row[key] is not None]
            item["mean_" + key] = _mean(values)
            item["sd_across_split_means_" + key] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        summary.append(item)
    return per_split, summary


def print_summary(summary):
    print("\nNB301 BEHAVIOR SUMMARY | feasibility, then anytime error, then final error")
    for budget in sorted({row["query_budget"] for row in summary}):
        members = [row for row in summary if row["query_budget"] == budget]
        members.sort(
            key=lambda row: (
                -row["valid_runs"] / row["model_runs"],
                float("inf")
                if row["mean_anytime_mean_error"] is None
                else row["mean_anytime_mean_error"],
                float("inf") if row["mean_best_error"] is None else row["mean_best_error"],
            )
        )
        label = "guarded exhaustion" if budget == -1 else f"{budget} terminal queries"
        print(f"\n{label}")
        print(
            "#  Algorithm                      Valid   Anytime   Best    Queries  Scored  RSS+MiB"
        )
        for rank, row in enumerate(members, 1):
            anytime = row["mean_anytime_mean_error"]
            best = row["mean_best_error"]
            rss = row["mean_rss_peak_delta_bytes"]
            print(
                f"{rank:<2} {row['algorithm']:<30} "
                f"{row['valid_runs']:>2}/{row['model_runs']:<2} "
                f"{anytime if anytime is not None else float('nan'):>8.3f} "
                f"{best if best is not None else float('nan'):>7.3f} "
                f"{row['mean_queries']:>9.1f} {row['mean_scored']:>7.1f} "
                f"{rss / (1024 * 1024) if rss is not None else float('nan'):>8.1f}"
            )


def save_csv(path, rows):
    scalars = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    if not scalars:
        return
    columns = list(dict.fromkeys(key for row in scalars for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(scalars)
