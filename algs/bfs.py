"""Plain FIFO breadth-first traversal; it does not consult the heuristic."""

from collections import deque

from .common import Budget, Run


def bfs(problem, *, budget=Budget(), event=None):
    run = Run(problem, budget, event)
    pending = deque([(problem.initial, 0)])
    while pending and not run.stopped:
        state, depth = pending.popleft()
        children = run.visit(state, depth)
        if children is not None:
            run.emit("retain", state=state, b=len(children), k=len(children), confidence=None)
            pending.extend((child, depth + 1) for child in children)
        run.pressure(len(pending))
    return run.result
