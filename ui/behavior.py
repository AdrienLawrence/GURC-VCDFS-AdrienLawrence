"""Interactive read-only view of an aggregated V-family behavior sweep."""

import math
from pathlib import Path

from schemas import load_payload

RESOURCE_METRICS = {
    "scores": ("mean_scored", "Mean heuristic evaluations"),
    "expansions": ("mean_expanded", "Mean expanded states"),
    "runtime": ("mean_runtime_seconds", "Mean runtime (seconds)"),
    "rss": ("mean_rss_peak_delta_bytes", "Mean RSS increase (bytes)"),
    "python-memory": ("mean_python_peak_bytes", "Mean Python allocation peak (bytes)"),
    "pending": ("mean_peak_pending_entries", "Mean peak pending states"),
}
QUALITY_METRICS = {
    "anytime": ("mean_anytime_mean_regret_pp", "Anytime mean regret (pp)"),
    "final": ("mean_final_regret_pp", "Final regret (pp)"),
}
FAMILIES = {
    "Baselines": "baseline",
    "VCDFS": "vcdfs",
    "VCDFS-A": "vcdfsa",
    "VCBFS": "vcbfs",
    "VCBFS-A": "vcbfsa",
    "VCBFSN": "vcbfsn",
    "VCBFSN-A": "vcbfsna",
}


def belongs_to_family(row, family):
    """Match one summary row to a viewer group."""
    implementation = row["implementation"]
    if FAMILIES[family] == "baseline":
        return implementation not in {"vcdfs", "vcdfsa", "vcbfs", "vcbfsa", "vcbfsn", "vcbfsna"}
    return implementation == FAMILIES[family]


def load_behavior_summary(results):
    path = Path(results)
    if path.is_dir():
        path = path / "summary.json"
    rows = load_payload(path, "behavior_summary")
    required = {
        "algorithm",
        "implementation",
        "query_budget",
        "mean_final_regret_pp",
        "mean_anytime_mean_regret_pp",
        "mean_all_optima_pruned",
        "mean_optimum_found",
    }
    if not isinstance(rows, list) or not rows:
        raise ValueError("Behavior summary must contain rows")
    if any(not isinstance(row, dict) or not required.issubset(row) for row in rows):
        raise ValueError("Malformed behavior summary row")
    return rows


def nondominated(rows, resource_key, quality_key):
    candidates = [
        row
        for row in rows
        if all(
            math.isfinite(float(row[key])) and float(row[key]) >= 0
            for key in (resource_key, quality_key)
        )
    ]
    return sorted(
        (
            row
            for row in candidates
            if not any(
                float(other[resource_key]) <= float(row[resource_key])
                and float(other[quality_key]) <= float(row[quality_key])
                and (
                    float(other[resource_key]) < float(row[resource_key])
                    or float(other[quality_key]) < float(row[quality_key])
                )
                for other in candidates
            )
        ),
        key=lambda row: (float(row[resource_key]), row["algorithm"]),
    )


def show_behavior(rows, initial_budget=None, initial_family="VCDFS"):
    """Open a Matplotlib explorer without writing or mutating artifacts."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.widgets import RadioButtons

    budgets = sorted({int(row["query_budget"]) for row in rows})
    budget = budgets[-1] if initial_budget is None else initial_budget
    if budget not in budgets:
        raise ValueError(f"Budget {budget} is absent from the summary")
    if initial_family not in FAMILIES:
        raise ValueError(f"Unknown family: {initial_family}")

    available_resources = {
        name: definition
        for name, definition in RESOURCE_METRICS.items()
        if all(definition[0] in row for row in rows)
    }
    state = {"budget": budget, "family": initial_family, "resource": "scores", "quality": "anytime"}
    figure, axes = plt.subplots(figsize=(15.5, 8.5))
    figure.subplots_adjust(left=0.20, right=0.73, bottom=0.13, top=0.90)
    family_axes = figure.add_axes((0.015, 0.65, 0.145, 0.22))
    budget_axes = figure.add_axes((0.015, 0.32, 0.145, 0.29))
    resource_axes = figure.add_axes((0.015, 0.12, 0.145, 0.17))
    quality_axes = figure.add_axes((0.015, 0.02, 0.145, 0.08))
    key_axes = figure.add_axes((0.75, 0.10, 0.24, 0.80))
    family_selector = RadioButtons(
        family_axes, tuple(FAMILIES), active=tuple(FAMILIES).index(initial_family)
    )
    budget_labels = tuple("Exhaustive" if value == -1 else str(value) for value in budgets)
    budget_lookup = dict(zip(budget_labels, budgets, strict=True))
    budget_selector = RadioButtons(budget_axes, budget_labels, active=budgets.index(budget))
    resource_selector = RadioButtons(resource_axes, tuple(available_resources))
    quality_selector = RadioButtons(quality_axes, tuple(QUALITY_METRICS))
    prune_map = LinearSegmentedColormap.from_list("pruning", ("#16855b", "#f0ad31", "#b83a3a"))

    def draw():
        resource_key, resource_label = available_resources[state["resource"]]
        quality_key, quality_label = QUALITY_METRICS[state["quality"]]
        selected = sorted(
            (
                row
                for row in rows
                if int(row["query_budget"]) == state["budget"]
                and belongs_to_family(row, state["family"])
            ),
            key=lambda row: row["algorithm"],
        )
        if not selected:
            raise ValueError("Selected family/budget has no rows")
        frontier = nondominated(selected, resource_key, quality_key)
        frontier_names = {row["algorithm"] for row in frontier}
        identifiers = {row["algorithm"]: index for index, row in enumerate(selected, start=1)}
        axes.clear()
        key_axes.clear()
        for row in selected:
            name = row["algorithm"]
            prune_rate = float(row["mean_all_optima_pruned"])
            found_rate = float(row["mean_optimum_found"])
            axes.scatter(
                float(row[resource_key]),
                float(row[quality_key]),
                s=70 + 130 * found_rate,
                color=prune_map(prune_rate),
                edgecolor="#102e25" if name in frontier_names else "white",
                linewidth=2.2 if name in frontier_names else 0.8,
                zorder=3,
            )
            axes.annotate(
                str(identifiers[name]),
                (float(row[resource_key]), float(row[quality_key])),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=4,
            )
        if frontier:
            axes.plot(
                [float(row[resource_key]) for row in frontier],
                [float(row[quality_key]) for row in frontier],
                color="#173f35",
                linewidth=2,
                zorder=2,
            )
        resource_values = [float(row[resource_key]) for row in selected]
        if all(value > 0 for value in resource_values):
            axes.set_xscale("log")
            axes.set_xlim(min(resource_values) / 1.25, max(resource_values) * 1.25)
        axes.invert_yaxis()
        axes.set_xlabel(f"{resource_label} — logarithmic scale (left is better)")
        axes.set_ylabel(f"{quality_label} (top is better)")
        budget_text = (
            "exhaustive search" if state["budget"] == -1 else f"query budget {state['budget']}"
        )
        axes.set_title(f"{state['family']} parameter behavior at {budget_text}")
        axes.grid(which="both", alpha=0.22)

        key_axes.set_axis_off()
        key_axes.text(0, 1.0, "Configuration key", fontsize=12, fontweight="bold", va="top")
        key_axes.text(
            0,
            0.955,
            "Dark outline/line: Pareto frontier\n"
            "Green → red: 0% → 100% all-optimum pruning\n"
            "Larger point: optimum found more often",
            fontsize=8.5,
            color="#44504c",
            va="top",
            linespacing=1.5,
        )
        for index, row in enumerate(selected):
            name = row["algorithm"]
            key_axes.text(
                0,
                0.82 - index * 0.052,
                f"{identifiers[name]:>2}  {name}",
                fontsize=8.2,
                color="#173f35" if name in frontier_names else "#59635f",
                va="top",
            )
        figure.canvas.draw_idle()

    def select_family(label):
        state["family"] = label
        draw()

    def select_budget(label):
        state["budget"] = budget_lookup[label]
        draw()

    def select_resource(label):
        state["resource"] = label
        draw()

    def select_quality(label):
        state["quality"] = label
        draw()

    family_selector.on_clicked(select_family)
    budget_selector.on_clicked(select_budget)
    resource_selector.on_clicked(select_resource)
    quality_selector.on_clicked(select_quality)
    draw()
    plt.show()
