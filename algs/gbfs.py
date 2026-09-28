"""Deterministic Greedy Best-First Search over the shared problem contract."""

from heapq import heappop, heappush
from itertools import count

from .common import Budget, Run


def gbfs(problem, *, budget=Budget(), event=None):
    """Always expand the globally pending state with smallest heuristic value.

    GBFS retains every generated state. It is greedy in expansion order, not a
    pruning algorithm, and path cost does not participate in its priority.
    """
    run = Run(problem, budget, event)
    ties = count()
    frontier = [(0.0, next(ties), problem.initial, 0)]
    while frontier and not run.stopped:
        score, _, state, depth = heappop(frontier)
        run.emit("gbfs_pop", state=state, depth=depth, score=score)
        children = run.visit(state, depth)
        if children is not None:
            for child_score, child in run.rank(children):
                heappush(frontier, (child_score, next(ties), child, depth + 1))
        run.pressure(len(frontier))
    return run.result
