"""Dynamic Beam Search adapted from Merenda et al. (2020).

The paper consumes decoder probabilities. Generic optimization problems expose
heuristic costs instead, so this adapter explicitly converts lower-is-better
costs to a probability distribution with a temperature-scaled softmax.
"""

from dataclasses import dataclass
from math import exp, floor, isfinite, log, sqrt

from .common import Budget, Run, finite, positive_integer


def costs_to_probabilities(costs, temperature):
    """Convert finite lower-is-better costs to normalized probabilities."""
    if not isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = [finite(value) for value in costs]
    if not values:
        return []
    best = min(values)
    weights = [exp(-(value - best) / temperature) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def shannon_entropy(probabilities):
    """Natural-log Shannon entropy, as used by the paper's first policy."""
    return -sum(p * log(p) for p in probabilities if p > 0)


def top_k_standard_deviation(probabilities, top_k):
    """Population standard deviation of the largest top-k probabilities."""
    positive_integer(top_k, "top_k")
    if not probabilities:
        return 0.0
    selected = sorted(probabilities, reverse=True)[:top_k]
    mean = sum(selected) / len(selected)
    return sqrt(sum((value - mean) ** 2 for value in selected) / len(selected))


@dataclass(frozen=True)
class DynamicBeamConfig:
    """Published linear width controller plus the explicit NAS adapter scale."""

    policy: str
    minimum_width: int
    maximum_width: int
    slope: float
    intercept: float
    top_k: int | None = None
    temperature: float = 1.0

    def __post_init__(self):
        if self.policy not in {"entropy", "standard_deviation"}:
            raise ValueError("policy must be 'entropy' or 'standard_deviation'")
        positive_integer(self.minimum_width, "minimum_width")
        positive_integer(self.maximum_width, "maximum_width")
        if self.minimum_width > self.maximum_width:
            raise ValueError("minimum_width cannot exceed maximum_width")
        if not isfinite(float(self.slope)) or not isfinite(float(self.intercept)):
            raise ValueError("slope and intercept must be finite")
        if not isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.policy == "standard_deviation":
            if self.top_k is None:
                raise ValueError("top_k is required for standard_deviation")
            positive_integer(self.top_k, "top_k")
        elif self.top_k is not None:
            raise ValueError("top_k is only valid for standard_deviation")

    def width(self, costs):
        probabilities = costs_to_probabilities(costs, self.temperature)
        if not probabilities:
            return 0, 0.0
        if self.policy == "entropy":
            statistic = shannon_entropy(probabilities)
        else:
            statistic = top_k_standard_deviation(probabilities, self.top_k)
        # The paper specifies nearest-integer linear mapping. Half-up is explicit
        # here because Python's built-in round uses ties-to-even.
        mapped = floor(self.slope * statistic + self.intercept + 0.5)
        width = min(self.maximum_width, max(self.minimum_width, mapped))
        return min(len(costs), width), statistic


def dynamic_beam(problem, config, *, budget=Budget(), event=None):
    """Run global level-synchronous beam search with a dynamic beam width."""
    if not isinstance(config, DynamicBeamConfig):
        raise TypeError("config must be a DynamicBeamConfig")
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
        width, statistic = config.width([score for score, _, _ in candidates])
        run.result.pruned += max(0, len(candidates) - width)
        run.emit(
            "dynamic_beam",
            policy=config.policy,
            statistic=statistic,
            b=len(candidates),
            k=width,
        )
        level = [(state, depth) for _, state, depth in candidates[:width]]
    return run.result
