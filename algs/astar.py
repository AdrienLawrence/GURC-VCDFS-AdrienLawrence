"""Deterministic tree-search A* with unit edge costs."""

from heapq import heappop, heappush
from itertools import count

from .common import Budget, Run


def astar(problem, *, budget=Budget(), event=None):
    """Prioritize ``depth + heuristic(state)`` without pruning.

    The shared search contract has no domain path-cost function, so every edge
    has cost one. This is an explicit control, not a claim that the transformer
    error estimate is an admissible A* heuristic.
    """
    run = Run(problem, budget, event)
    ties = count()
    frontier = [(0.0, next(ties), problem.initial, 0)]
    while frontier and not run.stopped:
        priority, _, state, depth = heappop(frontier)
        run.emit("astar_pop", state=state, depth=depth, priority=priority)
        children = run.visit(state, depth)
        if children is not None:
            for score, child in run.rank(children):
                heappush(frontier, (depth + 1 + score, next(ties), child, depth + 1))
        run.pressure(len(frontier))
    return run.result
