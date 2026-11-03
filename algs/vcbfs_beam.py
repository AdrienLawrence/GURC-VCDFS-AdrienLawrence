"""Historical pooled-frontier implementation formerly labeled VCBFS.

This is an adaptive global beam: it computes confidence and retention width on
the same pooled next-level candidate set. It is preserved for reproducibility,
but it is not the parent-local VCBFS algorithm.
"""

from .common import Budget, Run
from .vcbfs import VCBFSConfig, breadth_confidence


def vcbfs_beam(problem, pmin=1, pmax=10, *, max_beam=None, epsilon=1e-5, **kwargs):
    config = VCBFSConfig(pmin, pmax, "proportional", max_beam, epsilon)
    return _vcbfs_beam(problem, config, **kwargs)


def vcbfs_beama(problem, kmin=1, kmax=10, *, max_beam=None, epsilon=1e-5, **kwargs):
    config = VCBFSConfig(kmin, kmax, "absolute", max_beam, epsilon)
    return _vcbfs_beam(problem, config, **kwargs)


def _vcbfs_beam(problem, config, *, budget=Budget(), event=None):
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
        candidates.sort(key=lambda item: item[0])
        if candidates:
            confidence = breadth_confidence([score for score, _, _ in candidates], config.epsilon)
            width = config.width(len(candidates), confidence)
        else:
            confidence, width = None, 0
        run.result.pruned += max(0, len(candidates) - width)
        run.emit("vcbfs_beam", confidence=confidence, b=len(candidates), k=width)
        level = [(state, depth) for _, state, depth in candidates[:width]]
    return run.result
