"""Breadth-first search with VCDFS-style sibling-local normalization."""

from .common import Budget, Run
from .vcdfs import Retention


def vcbfsn(problem, pmin=20, pmax=100, *, epsilon=1e-5, **kwargs):
    retention = Retention("proportional", pmin, pmax, epsilon)
    return _vcbfsn(problem, retention, **kwargs)


def vcbfsna(problem, kmin=1, kmax=3, *, epsilon=1e-5, **kwargs):
    retention = Retention("absolute", kmin, kmax, epsilon)
    return _vcbfsn(problem, retention, **kwargs)


def _vcbfsn(problem, retention, *, budget=Budget(), event=None):
    """Apply confidence and retention independently to every sibling group."""
    run = Run(problem, budget, event)
    level = [(problem.initial, 0)]
    while level and not run.stopped:
        next_level = []
        event_groups = []
        for position, (parent, depth) in enumerate(level):
            children = run.visit(parent, depth)
            if children is not None:
                ranked = run.rank(children)
                scores = [score for score, _ in ranked]
                width, local_confidence = retention.width(scores)
                retained, pruned = ranked[:width], ranked[width:]
                run.result.pruned += len(pruned)
                next_level.extend((state, depth + 1) for _, state in retained)
                event_groups.append(
                    {
                        "parent": parent,
                        "b": len(ranked),
                        "k": width,
                        "confidence": local_confidence,
                        "retained": [{"state": state, "score": score} for score, state in retained],
                        "pruned": [{"state": state, "score": score} for score, state in pruned],
                    }
                )
            run.pressure(len(level) - position - 1 + len(next_level))
            if run.stopped:
                break
        if run.stopped:
            break
        run.emit(
            "vcbfsn",
            active_parent_groups=len(event_groups),
            total_candidates=sum(group["b"] for group in event_groups),
            groups=event_groups,
        )
        level = next_level
        run.pressure(len(level))
    return run.result
