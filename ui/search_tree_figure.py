"""Render an already captured root-to-leaf explanation; no search imports."""

import statistics


def _node_text(data, prefix, prediction):
    depth = len(prefix)

    if depth == 0:
        edge = data.edges[0]
        return (
            "ROOT: empty architecture\n"
            f"$\\mathbf{{Next:~edge~{edge[0]}\\to{edge[1]}}}$\n"
            f"h={prediction:.3f}%\n"
            f"visit {data.visit_order[prefix]}"
        )

    operation = data.operations[prefix[-1]]
    decided = data.edges[depth - 1]

    if depth < len(data.edges):
        following = data.edges[depth]
        return (
            f"edge {decided[0]}→{decided[1]} = {operation}\n"
            f"$\\mathbf{{Next:~edge~{following[0]}\\to{following[1]}}}$\n"
            f"h={prediction:.3f}%  ·  visit {data.visit_order[prefix]}"
        )

    return (
        f"edge {decided[0]}→{decided[1]} = {operation}\n"
        f"leaf completion\n"
        f"h={prediction:.3f}%  ·  visit {data.visit_order[prefix]}"
    )


def _decision_statistics(decision, lower, upper, proportional):
    mean = statistics.fmean(decision.confidence_scores)
    mad = statistics.fmean(abs(score - mean) for score in decision.confidence_scores)
    gap = decision.confidence_scores[1] - decision.confidence_scores[0]
    q = decision.confidence * lower + (1 - decision.confidence) * upper
    raw_width = len(decision.candidates) * q / 100 if proportional else q
    return mean, mad, gap, q, raw_width


def _wrap_architecture(architecture, width=42):
    """Wrap a NAS-Bench-201 architecture at operation boundaries without changing content."""
    cells = architecture.split("+")
    lines = []

    for cell_index, cell in enumerate(cells):
        suffix = "+" if cell_index < len(cells) - 1 else ""
        tokens = [token for token in cell.split("|") if token]
        current = ""

        for token in tokens:
            piece = f"|{token}"
            if current and len(current) + len(piece) + 1 > width:
                lines.append(current + "|")
                current = piece
            else:
                current += piece

        if current:
            lines.append(current + "|" + suffix)

    return "\n".join(lines)


def _draw_path(tree, data, colors):
    path_length = len(data.path)

    for depth in range(path_length):
        tree.plot(
            (depth, depth + 1),
            (0, 0),
            color=colors["path"],
            linewidth=3.2,
            zorder=1,
        )

    for depth in range(path_length + 1):
        prefix = data.path[:depth]

        tree.scatter(
            depth,
            0,
            s=560,
            color=colors["node"],
            edgecolor=colors["path"],
            linewidth=2.4,
            zorder=3,
        )

        if depth < path_length:
            tree.annotate(
                _node_text(data, prefix, data.path_predictions[depth]),
                (depth, 0),
                xytext=(0, 35 if depth % 2 == 0 else -52),
                textcoords="offset points",
                ha="center",
                va="bottom" if depth % 2 == 0 else "top",
                fontsize=8.2,
                zorder=5,
            )

    final_edge = data.edges[-1]
    final_operation = data.operations[data.path[-1]]

    tree.annotate(
        f"FINAL: edge {final_edge[0]}→{final_edge[1]} = {final_operation}\n"
        f"LEAF — complete architecture\n"
        f"{_wrap_architecture(data.architecture)}\n"
        f"transformer h(t)={data.path_predictions[-1]:.3f}%\n"
        f"NAS measured J(t)={data.measured_error:.3f}%\n"
        f"prediction error={data.path_predictions[-1] - data.measured_error:+.3f} pp\n"
        f"terminal query {data.terminal_query}",
        (len(data.path), 0),
        xytext=(12, 34),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.2,
        zorder=5,
    )


def _child_label(data, child, score, kept):
    depth = len(child) - 1
    decided = data.edges[depth]
    operation = data.operations[child[-1]]
    status = "KEEP" if kept else "DISCARD"

    if len(child) < len(data.edges):
        following = data.edges[len(child)]
        return (
            f"{operation}\n"
            f"edge {decided[0]}→{decided[1]}\n"
            f"$\\mathbf{{Next:~edge~{following[0]}\\to{following[1]}}}$\n"
            f"h={score:.3f}% · {status}"
        )

    return (
        f"{operation}\nedge {decided[0]}→{decided[1]}\nleaf completion\nh={score:.3f}% · {status}"
    )


def _draw_branches(tree, data, colors):
    highlighted = {decision.parent for decision in data.panel_decisions}

    for decision in data.all_decisions:
        depth = len(decision.parent)
        path_child = data.path[: depth + 1]
        children = [item for item in decision.candidates if item[1] != path_child]
        retained = {state for _, state in decision.retained}

        sign = 1 if depth % 2 == 0 else -1
        base = 0.52 if depth in (0, 5) else 0.58
        span = 1.15 if len(children) <= 2 else 1.55
        offsets = [
            sign * (base + span * index / max(1, len(children) - 1))
            for index in range(len(children))
        ]

        branch_length = 0.70 if depth < 5 else 0.62
        line_width = 1.8 if decision.parent in highlighted else 1.3

        for (score, child), y in zip(children, offsets, strict=True):
            kept = child in retained
            color = colors["retained"] if kept else colors["pruned"]

            tree.plot(
                (depth, depth + branch_length),
                (0, y),
                color=color,
                linewidth=line_width,
                zorder=1,
            )

            tree.scatter(
                depth + branch_length,
                y,
                s=235,
                color="#e8f4ef" if kept else "#eeeeee",
                edgecolor=color,
                linewidth=1.6,
                marker="o" if kept else "X",
                zorder=3,
            )

            tree.annotate(
                _child_label(data, child, score, kept),
                (depth + branch_length, y),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.2,
                color="#174b3c" if kept else "#666c69",
            )


def _draw_decision_panel(axes, data, decision, label):
    axes.axis("off")

    proportional = not data.algorithm.endswith("a")
    mean, mad, gap, q, raw = _decision_statistics(
        decision,
        data.lower,
        data.upper,
        proportional,
    )

    retained = {state for _, state in decision.retained}

    ranking = [
        (
            f"h{rank}={score:.3f} "
            f"({data.operations[state[-1]]}, "
            f"{'KEEP' if state in retained else 'DISCARD'})"
        )
        for rank, (score, state) in enumerate(decision.candidates, start=1)
    ]

    ranking_lines = "\n".join(f"  {item}" for item in ranking)

    if data.algorithm.startswith("vcbfs"):
        scope = f"pooled level population n={len(decision.confidence_scores)}"
        if mad < data.epsilon:
            equation = f"c=0.5 plateau because MAD<ε ({mad:.4f}<{data.epsilon:g})"
        else:
            equation = f"c=clip({gap:.4f}/{mad:.4f},0,1)={decision.confidence:.4f}"
    else:
        scope = f"all b={len(decision.candidates)} siblings"
        equation = f"c=clip({gap:.4f}/({mad:.4f}+{data.epsilon:g}),0,1)={decision.confidence:.4f}"

    axes.text(
        0,
        1,
        f"{label} — parent depth {len(decision.parent)}\n"
        f"Confidence population: {scope}\n"
        f"Sorted local children:\n"
        f"{ranking_lines}\n"
        f"h₂−h₁={gap:.4f}; mean h={mean:.4f}; MAD={mad:.4f}\n"
        f"{equation}\n"
        f"q(c)=c·{data.lower:g}+(1−c)·{data.upper:g}={q:.3f}"
        f"{'%' if proportional else ' children'}; "
        f"raw k={raw:.3f}; actual k={len(decision.retained)}",
        ha="left",
        va="top",
        fontsize=8.25,
        linespacing=1.28,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f6f8f7",
            "edgecolor": "#cad2ce",
        },
    )


def show_search_tree(data):
    """Render one real-data explanatory figure; never save automatically."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure = plt.figure(figsize=(19.8, 11.2))
    tree = figure.add_axes((0.035, 0.42, 0.93, 0.51))
    panels = [
        figure.add_axes((0.04, 0.065, 0.445, 0.30)),
        figure.add_axes((0.515, 0.065, 0.445, 0.30)),
    ]

    tree.set(
        xlim=(-0.35, len(data.path) + 2.05),
        ylim=(-2.45, 2.45),
    )
    tree.axis("off")

    colors = {
        "path": "#1464a0",
        "retained": "#16855b",
        "pruned": "#a4aaa7",
        "node": "#f7faf9",
    }

    _draw_path(tree, data, colors)
    _draw_branches(tree, data, colors)

    tree.set_title(
        "One real NAS-Bench-201 search: empty architecture → measured complete architecture",
        fontsize=15,
        pad=18,
    )

    tree.legend(
        handles=[
            Line2D(
                (0,),
                (0,),
                color=colors["path"],
                linewidth=3,
                label="selected retained path",
            ),
            Line2D(
                (0,),
                (0,),
                marker="o",
                color=colors["retained"],
                linewidth=0,
                label="retained",
            ),
            Line2D(
                (0,),
                (0,),
                marker="X",
                color=colors["pruned"],
                linewidth=0,
                label="discarded",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    labels = ("LOWER observed confidence", "HIGHER observed confidence")
    for panel, decision, label in zip(
        panels,
        data.panel_decisions,
        labels,
        strict=True,
    ):
        _draw_decision_panel(panel, data, decision, label)

    mode = "proportional" if not data.algorithm.endswith("a") else "absolute"

    figure.text(
        0.5,
        0.015,
        (
            f"Figure. {data.algorithm.upper()} on NAS-Bench-201 v1.1, "
            "CIFAR-10-valid x-valid validation error at 200 epochs; "
            f"held-out split seed {data.split_seed}, transformer model seed "
            f"{data.model_seed}; {mode} bounds [{data.lower:g}, {data.upper:g}], "
            f"ε={data.epsilon:g}. Internal h(s) values are frozen-transformer "
            "estimates for completions of each prefix; leaf J(t) is the NAS-Bench-201 "
            "available-trial mean measured validation error. All expanded parents on the "
            "displayed path show retained/discarded children; the two panels give the "
            "lowest- and highest-confidence path decisions in full detail."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )

    output = "artifacts/vcdfs_search_tree.svg"

    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
    )

    print(f"Saved SVG: {output}")

    plt.show()
