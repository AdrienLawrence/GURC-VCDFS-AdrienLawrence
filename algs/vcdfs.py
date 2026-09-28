"""Clipped all-sibling confidence, with explicit proportional/absolute limits.

This is the September assignment specification. It is not a calibrated
probability and does not make pruning safe. See PROTOCOL.md for manuscript caveats.
"""

from dataclasses import dataclass
from math import floor, fsum, isfinite

from .common import finite
from .dfs import dfs


def confidence(scores, epsilon=1e-5):
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    values = sorted(finite(x) for x in scores)
    if len(values) < 2:
        return None
    mean = fsum(x / len(values) for x in values)
    mad = fsum(abs(x - mean) / len(values) for x in values)
    gap = values[1] - values[0]
    if not isfinite(mad) or not isfinite(gap) or not isfinite(mad + epsilon):
        raise ValueError("Score magnitudes overflow confidence arithmetic")
    return min(1.0, max(0.0, gap / (mad + epsilon)))


@dataclass(frozen=True)
class Retention:
    mode: str = "proportional"
    lower: float = 20
    upper: float = 100
    epsilon: float = 1e-5

    def __post_init__(self):
        if self.mode not in {"proportional", "absolute", "fixed", "all"}:
            raise ValueError("Unknown retention mode")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if any(isinstance(x, bool) or not isfinite(x) for x in (self.lower, self.upper)):
            raise ValueError("Bounds must be finite numbers, not booleans")
        if self.lower > self.upper:
            raise ValueError("Lower bound exceeds upper bound")
        if self.mode == "proportional" and not 0 <= self.lower <= self.upper <= 100:
            raise ValueError("Percentages must lie in [0, 100]")
        if self.mode in {"absolute", "fixed"}:
            if self.lower < 1 or any(int(x) != x for x in (self.lower, self.upper)):
                raise ValueError("Absolute bounds must be positive integers")
        if self.mode == "fixed" and self.lower != self.upper:
            raise ValueError("Fixed retention requires equal bounds")

    def width(self, scores):
        values = [finite(x) for x in scores]
        b = len(values)
        if b < 2 or self.mode == "all":
            return b, None
        if self.mode == "fixed":
            return min(b, int(self.lower)), None
        conf = confidence(values, self.epsilon)
        # Convex combination avoids overflowing upper-lower at large bounds.
        raw = conf * self.lower + (1 - conf) * self.upper
        if self.mode == "proportional":
            raw *= b / 100
        return min(b, max(1, floor(raw))), conf


def vcdfs(problem, pmin=20, pmax=100, *, epsilon=1e-5, **kwargs):
    return dfs(problem, selector=Retention("proportional", pmin, pmax, epsilon).width, **kwargs)


def vcdfsa(problem, kmin=1, kmax=3, *, epsilon=1e-5, **kwargs):
    return dfs(problem, selector=Retention("absolute", kmin, kmax, epsilon).width, **kwargs)
