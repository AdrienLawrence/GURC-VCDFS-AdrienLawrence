"""Interactive read-only Pareto view for a completed search summary."""

import math
from pathlib import Path

from schemas import load_payload

RESOURCE_METRICS = {
    "cost": (
        "memory_time_product",
        "Runtime × RSS increase (MiB·seconds)",
        1.0,
    ),
    "runtime": ("mean_runtime_seconds", "Mean runtime (seconds)", 1.0),
    "rss": ("mean_rss_peak_delta_bytes", "Mean RSS increase (MiB)", 1024**2),
    "queries": ("mean_queries", "Mean terminal queries", 1.0),
    "scores": ("mean_scored", "Mean heuristic evaluations", 1.0),
    "expansions": ("mean_expanded", "Mean expansions", 1.0),
    "pending": ("mean_peak_pending_entries", "Mean peak pending states", 1.0),
}


def resource_value(row, resource_key):
    """Read a resource measure, including the derived memory-time product."""
    if resource_key == "memory_time_product":
        runtime = row.get("mean_runtime_seconds")
        memory = row.get("mean_rss_peak_delta_bytes")
        if runtime is None or memory is None:
            return None
        return float(runtime) * float(memory) / (1024**2)
    value = row.get(resource_key)
    return None if value is None else float(value)


def load_summary(results):
    """Load a version-2 search summary without modifying its source artifact."""
    path = Path(results)
    if path.is_dir():
        path = path / "summary.json"
    rows = load_payload(path, "search_summary")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Search summary must contain at least one algorithm row")
    required = {"algorithm", "mean_regret_pp"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Malformed search-summary row")
    return rows


def pareto_rows(rows, resource_key):
    """Return configurations nondominated when both resource and regret are minimized."""
    candidates = []
    for row in rows:
        resource, regret = resource_value(row, resource_key), row.get("mean_regret_pp")
        if resource is None or regret is None:
            continue
        resource, regret = float(resource), float(regret)
        if math.isfinite(resource) and math.isfinite(regret):
            candidates.append((resource, regret, row))
    frontier = []
    for resource, regret, row in candidates:
        dominated = any(
            other_resource <= resource
            and other_regret <= regret
            and (other_resource < resource or other_regret < regret)
            for other_resource, other_regret, _ in candidates
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (resource_value(row, resource_key), row["algorithm"]))


def quality_marker_sizes(regrets, minimum=55.0, maximum=145.0):
    """Give better (lower-regret) results more visual weight."""
    numeric = [float(value) for value in regrets]
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise ValueError("Regrets must be finite")
    low, high = min(numeric), max(numeric)
    if low == high:
        return [(minimum + maximum) / 2] * len(numeric)
    return [maximum - (value - low) / (high - low) * (maximum - minimum) for value in numeric]


def show_pareto(rows, initial_metric="cost"):
    """Open an interactive Matplotlib window; never save or mutate artifacts."""
    if initial_metric not in RESOURCE_METRICS:
        raise ValueError(f"Unknown resource metric: {initial_metric}")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    figure, axes = plt.subplots(figsize=(16, 8))
    figure.subplots_adjust(left=0.18, right=0.71, bottom=0.13, top=0.9)
    selector_axes = figure.add_axes((0.015, 0.42, 0.13, 0.36))
    key_axes = figure.add_axes((0.73, 0.10, 0.26, 0.80))
    selector = RadioButtons(
        selector_axes, tuple(RESOURCE_METRICS), active=list(RESOURCE_METRICS).index(initial_metric)
    )

    def draw(metric):
        resource_key, label, divisor = RESOURCE_METRICS[metric]
        usable = [
            row
            for row in rows
            if resource_value(row, resource_key) is not None
            and row.get("mean_regret_pp") is not None
        ]
        frontier = pareto_rows(usable, resource_key)
        frontier_names = {row["algorithm"] for row in frontier}
        resources = [resource_value(row, resource_key) / divisor for row in usable]
        regrets = [float(row["mean_regret_pp"]) for row in usable]
        points = {
            row["algorithm"]: (x, y) for row, x, y in zip(usable, resources, regrets, strict=True)
        }
        axes.clear()
        key_axes.clear()
        ordered = sorted(usable, key=lambda row: row["algorithm"].casefold())
        identifiers = {row["algorithm"]: index for index, row in enumerate(ordered, start=1)}
        sizes = dict(
            zip(
                (row["algorithm"] for row in usable),
                quality_marker_sizes(regrets),
                strict=True,
            )
        )
        best_name = min(usable, key=lambda row: float(row["mean_regret_pp"]))["algorithm"]
        cheapest_name = min(usable, key=lambda row: resource_value(row, resource_key))["algorithm"]
        for row in ordered:
            name = row["algorithm"]
            color = "#11875d" if row["algorithm"] in frontier_names else "#8a9490"
            x, y = points[name]
            edgecolor = (
                "#d18b00" if name == best_name else "#1464a0" if name == cheapest_name else "white"
            )
            axes.scatter(
                x,
                y,
                color=color,
                edgecolor=edgecolor,
                linewidth=2.2 if name in {best_name, cheapest_name} else 0.8,
                s=sizes[name],
                zorder=3,
            )
            axes.annotate(
                str(identifiers[name]),
                (x, y),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=4,
            )
        if frontier:
            axes.plot(
                [points[row["algorithm"]][0] for row in frontier],
                [points[row["algorithm"]][1] for row in frontier],
                color="#11875d",
                linewidth=2,
                zorder=2,
            )
        positive_resources = [value for value in resources if value > 0]
        if len(positive_resources) == len(resources):
            axes.set_xscale("log")
            axes.set_xlim(min(positive_resources) / 1.35, max(positive_resources) * 1.35)
        elif positive_resources:
            axes.set_xscale("symlog", linthresh=min(positive_resources) / 10)
            axes.margins(x=0.08)
        else:
            axes.margins(x=0.08)
        regret_low, regret_high = min(regrets), max(regrets)
        regret_padding = max((regret_high - regret_low) * 0.04, 0.01)
        axes.set_ylim(regret_high + regret_padding, max(0.0, regret_low - regret_padding))
        axes.set_xlabel(f"{label} — logarithmic scale (left is better)")
        axes.set_ylabel("Mean search-only regret (percentage points; top is better)")
        axes.set_title("Search quality–resource Pareto frontier")
        axes.grid(which="both", alpha=0.22)

        key_axes.set_axis_off()
        key_axes.text(0, 1.0, "Algorithm key", fontsize=12, fontweight="bold", va="top")
        key_axes.text(
            0,
            0.955,
            "● Pareto frontier   ● Dominated\n"
            "Gold ring: best regret   Blue ring: least resource\n"
            "Larger points indicate lower regret",
            fontsize=8.5,
            color="#44504c",
            va="top",
            linespacing=1.5,
        )
        midpoint = (len(ordered) + 1) // 2
        for index, row in enumerate(ordered):
            column = 0 if index < midpoint else 1
            row_index = index if column == 0 else index - midpoint
            name = row["algorithm"]
            prefix = "◆" if name in frontier_names else "●"
            key_axes.text(
                column * 0.52,
                0.84 - row_index * 0.065,
                f"{identifiers[name]:>2}  {prefix} {name}",
                fontsize=8.1,
                color="#08724f" if name in frontier_names else "#59635f",
                va="top",
            )
        figure.canvas.draw_idle()

    selector.on_clicked(draw)
    draw(initial_metric)
    plt.show()
