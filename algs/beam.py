"""Fixed GLOBAL level beam, not a per-parent child limit."""

from .common import Budget, Run, positive_integer


def beam(problem, width=16, *, budget=Budget(), event=None):
    positive_integer(width, "width")
    run = Run(problem, budget, event)
    level = [(problem.initial, 0)]
    while level and not run.stopped:
        candidates = []
        for position, (state, depth) in enumerate(level):
            children = run.visit(state, depth)
            if children is not None:
                candidates.extend((score, child, depth + 1) for score, child in run.rank(children))
            run.pressure(len(level) - position - 1 + len(candidates))
            if run.stopped:
                break
        if run.stopped:
            break
        # Stable ties preserve parent order, then successor order.
        candidates.sort(key=lambda item: item[0])
        run.result.pruned += max(0, len(candidates) - width)
        run.emit("beam", b=len(candidates), k=min(width, len(candidates)))
        level = [(state, depth) for _, state, depth in candidates[:width]]
    return run.result
