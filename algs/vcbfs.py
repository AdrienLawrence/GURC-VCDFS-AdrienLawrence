"""Variable-Cluster Breadth-First Search with global level retention."""

from dataclasses import dataclass
from math import floor, fsum, isfinite

from .common import Budget, Run, finite, positive_integer


def breadth_confidence(scores, epsilon=1e-5):
    """Best/second gap normalized by level MAD, with reference plateau rule."""
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    values = sorted(finite(value) for value in scores)
    if not values:
        raise ValueError("breadth confidence requires at least one score")
    if len(values) == 1:
        return 1.0
    mean = fsum(value / len(values) for value in values)
    mad = fsum(abs(value - mean) / len(values) for value in values)
    if mad < epsilon:
        return 0.5
    return min(1.0, max(0.0, (values[1] - values[0]) / mad))


@dataclass(frozen=True)
class VCBFSConfig:
    """Bounds for the next global frontier; proportional is the default."""

    lower: float = 1
    upper: float = 10
    mode: str = "proportional"
    max_beam: int | None = None
    epsilon: float = 1e-5

    def __post_init__(self):
        if self.mode not in {"proportional", "absolute"}:
            raise ValueError("mode must be 'proportional' or 'absolute'")
        if any(isinstance(x, bool) or not isfinite(x) for x in (self.lower, self.upper)):
            raise ValueError("bounds must be finite numbers, not booleans")
        if self.lower > self.upper:
            raise ValueError("lower bound exceeds upper bound")
        if self.mode == "proportional" and not 0 <= self.lower <= self.upper <= 100:
            raise ValueError("proportional bounds must lie in [0, 100]")
        if self.mode == "absolute":
            if self.lower < 1 or any(int(x) != x for x in (self.lower, self.upper)):
                raise ValueError("absolute bounds must be positive integers")
        if self.max_beam is not None:
            positive_integer(self.max_beam, "max_beam")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    def width(self, branching, confidence):
        if branching < 1:
            return 0
        interpolated = self.lower + (1 - confidence) * (self.upper - self.lower)
        raw = branching * interpolated / 100 if self.mode == "proportional" else interpolated
        width = floor(raw + 0.5)
        if self.max_beam is not None:
            width = min(width, self.max_beam)
        return min(branching, max(1, width))


def vcbfs(problem, pmin=1, pmax=10, *, max_beam=None, epsilon=1e-5, **kwargs):
    config = VCBFSConfig(pmin, pmax, "proportional", max_beam, epsilon)
    return _vcbfs(problem, config, **kwargs)


def vcbfsa(problem, kmin=1, kmax=10, *, max_beam=None, epsilon=1e-5, **kwargs):
    config = VCBFSConfig(kmin, kmax, "absolute", max_beam, epsilon)
    return _vcbfs(problem, config, **kwargs)


def _vcbfs(problem, config, *, budget=Budget(), event=None):
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
        run.emit("vcbfs", confidence=confidence, b=len(candidates), k=width)
        level = [(state, depth) for _, state, depth in candidates[:width]]
    return run.result
