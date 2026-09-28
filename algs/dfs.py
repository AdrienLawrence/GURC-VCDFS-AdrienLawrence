"""Depth-first traversal. A selector changes pruning, not backtracking."""

from .common import Budget, Run


def dfs(problem, *, selector=None, budget=Budget(), event=None):
    run = Run(problem, budget, event)
    pending = [(problem.initial, 0)]
    while pending and not run.stopped:
        state, depth = pending.pop()
        children = run.visit(state, depth)
        if children is not None:
            conf = None
            if selector is None:
                selected = children
            else:
                ranked = run.rank(children)
                k, conf = selector([score for score, _ in ranked])
                if type(k) is not int or not 0 <= k <= len(children):
                    raise ValueError("Selector returned an invalid width")
                selected = [child for _, child in ranked[:k]]
            run.result.pruned += len(children) - len(selected)
            run.emit("retain", state=state, b=len(children), k=len(selected), confidence=conf)
            # Best child must be popped first; other retained siblings remain
            # pending while its entire subtree is explored. This IS backtracking.
            pending.extend((child, depth + 1) for child in reversed(selected))
        run.pressure(len(pending))
    return run.result


def heuristic_dfs(problem, **kwargs):
    return dfs(problem, selector=lambda scores: (len(scores), None), **kwargs)
