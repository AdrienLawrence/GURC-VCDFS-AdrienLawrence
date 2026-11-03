"""Capture a small, real NAS-Bench-201 trace for downstream explanation."""

from dataclasses import dataclass
from pathlib import Path

from algs import Budget, vcbfs, vcbfsa, vcdfs, vcdfsa
from bench201.heuristic.data import load_measurements
from bench201.heuristic.model import FrozenPredictor, checkpoint_metadata
from bench201.space import EDGES, OPERATIONS, ArchitectureTree, architecture_string
from schemas import load_payload

RUNNERS = {"vcdfs": vcdfs, "vcdfsa": vcdfsa, "vcbfs": vcbfs, "vcbfsa": vcbfsa}


@dataclass(frozen=True)
class Decision:
    parent: tuple
    confidence: float
    candidates: tuple
    retained: tuple
    pruned: tuple
    confidence_scores: tuple


@dataclass(frozen=True)
class FigureData:
    algorithm: str
    lower: float
    upper: float
    epsilon: float
    split_seed: int
    model_seed: int
    terminal_query: int
    path: tuple
    path_predictions: tuple
    measured_error: float
    all_decisions: tuple
    panel_decisions: tuple
    visit_order: dict
    edges: tuple = EDGES
    operations: tuple = OPERATIONS
    architecture: str = ""


def _decision_from_retain(event):
    retained = tuple((float(item["score"]), tuple(item["state"])) for item in event["retained"])
    pruned = tuple((float(item["score"]), tuple(item["state"])) for item in event["pruned"])
    candidates = tuple(sorted(retained + pruned, key=lambda item: item[0]))
    confidence = event.get("confidence")
    if confidence is None or len(candidates) < 2:
        return None
    return Decision(
        tuple(event["state"]),
        float(confidence),
        candidates,
        retained,
        pruned,
        tuple(score for score, _ in candidates),
    )


def decisions_from_events(events):
    """Recover actual candidate sets and decisions from algorithm events."""
    decisions = []

    for event in events:
        if event["kind"] == "retain":
            decision = _decision_from_retain(event)
            if decision is not None:
                decisions.append(decision)

        elif event["kind"] == "vcbfs" and event.get("confidence") is not None:
            pooled = tuple(
                float(item["score"])
                for group in event["groups"]
                for key in ("retained", "pruned")
                for item in group[key]
            )
            pooled = tuple(sorted(pooled))

            for group in event["groups"]:
                retained = tuple(
                    (float(item["score"]), tuple(item["state"])) for item in group["retained"]
                )
                pruned = tuple(
                    (float(item["score"]), tuple(item["state"])) for item in group["pruned"]
                )
                decisions.append(
                    Decision(
                        tuple(group["parent"]),
                        float(event["confidence"]),
                        tuple(sorted(retained + pruned, key=lambda item: item[0])),
                        retained,
                        pruned,
                        pooled,
                    )
                )

    return decisions


def select_explanatory_path(events, decisions):
    """Choose an observed terminal whose path contains the widest confidence contrast."""
    terminals = [tuple(event["state"]) for event in events if event["kind"] == "terminal"]
    if not terminals:
        raise RuntimeError("Search produced no terminal state")

    by_parent = {decision.parent: decision for decision in decisions}

    def contrast(path):
        values = [
            by_parent[path[:depth]].confidence for depth in range(6) if path[:depth] in by_parent
        ]
        return max(values) - min(values) if len(values) >= 2 else -1.0

    return max(terminals, key=contrast)


def collect_path_decisions(path, decisions):
    """Return all retention decisions encountered along the displayed path, ordered by depth."""
    by_parent = {decision.parent: decision for decision in decisions}
    ordered = []

    for depth in range(len(path)):
        parent = path[:depth]
        if parent in by_parent:
            ordered.append(by_parent[parent])

    return tuple(ordered)


def select_contrasting_decisions(path_decisions):
    """Pick two separated path parents with the largest observed confidence difference."""
    internal_only = [decision for decision in path_decisions if len(decision.parent) <= 4]
    relevant = internal_only if len(internal_only) >= 2 else list(path_decisions)

    if len(relevant) < 2:
        raise RuntimeError("At least two path retention decisions are required")

    separated = [
        (left, right)
        for index, left in enumerate(relevant)
        for right in relevant[index + 1 :]
        if abs(len(left.parent) - len(right.parent)) >= 2
    ]

    pairs = separated or [
        (left, right) for index, left in enumerate(relevant) for right in relevant[index + 1 :]
    ]

    low, high = max(pairs, key=lambda pair: abs(pair[0].confidence - pair[1].confidence))
    return (high, low) if low.confidence > high.confidence else (low, high)


def capture_figure_data(
    data_path,
    checkpoint,
    algorithm="vcdfs",
    lower=20,
    upper=100,
    epsilon=1e-5,
    query_budget=32,
):
    """Run one held-out search and return the minimal presentation data."""
    if algorithm not in RUNNERS:
        raise ValueError(f"Unknown V-family algorithm: {algorithm}")

    records, checksum = load_measurements(data_path)
    checkpoint = Path(checkpoint)
    metadata = checkpoint_metadata(checkpoint / "model.json")

    if metadata["data_sha256"] != checksum:
        raise ValueError("Checkpoint and measurement data checksums differ")

    split = load_payload(checkpoint / "split.json", "data_split", allow_legacy=True)
    objectives = {tuple(row["state"]): float(row["error"]) for row in records}
    allowed = tuple(tuple(records[index]["state"]) for index in split["test"])

    predictor = FrozenPredictor.load(checkpoint / "model.json")
    predictor(())

    events = []
    parameters = {"epsilon": epsilon}
    bound_names = ("kmin", "kmax") if algorithm.endswith("a") else ("pmin", "pmax")
    parameters.update({bound_names[0]: lower, bound_names[1]: upper})

    RUNNERS[algorithm](
        ArchitectureTree(predictor, objectives.__getitem__, allowed=allowed),
        budget=Budget(queries=query_budget),
        event=events.append,
        **parameters,
    )

    decisions = decisions_from_events(events)
    path = select_explanatory_path(events, decisions)
    path_decisions = collect_path_decisions(path, decisions)
    panel_decisions = select_contrasting_decisions(path_decisions)

    terminal_events = [event for event in events if event["kind"] == "terminal"]
    terminal_query = next(
        index
        for index, event in enumerate(terminal_events, start=1)
        if tuple(event["state"]) == path
    )

    visit_order = {}
    for index, event in enumerate((event for event in events if event["kind"] == "visit"), start=1):
        visit_order.setdefault(tuple(event["state"]), index)

    return FigureData(
        algorithm=algorithm,
        lower=float(lower),
        upper=float(upper),
        epsilon=float(epsilon),
        split_seed=int(metadata["split_seed"]),
        model_seed=int(metadata["model_seed"]),
        terminal_query=terminal_query,
        path=path,
        path_predictions=tuple(float(predictor(path[:depth])) for depth in range(7)),
        measured_error=objectives[path],
        all_decisions=path_decisions,
        panel_decisions=panel_decisions,
        visit_order=visit_order,
        architecture=architecture_string(path),
    )
