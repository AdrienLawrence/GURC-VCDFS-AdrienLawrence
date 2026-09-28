"""Shared accounting, not a shared traversal algorithm.

Contract: immutable states, deterministic finite construction TREE, lower scores
are better. No cycle detection, graph dominance, or optimality guarantee.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Protocol


class Problem(Protocol):
    initial: Any

    def successors(self, state: Any) -> Iterable[Any]: ...
    def feasible(self, state: Any) -> bool: ...
    def terminal(self, state: Any) -> bool: ...
    def heuristic(self, state: Any) -> float: ...
    def objective(self, state: Any) -> float: ...


def finite(value):
    value = float(value)
    if not isfinite(value):
        raise ValueError("Heuristics and objectives must be finite")
    return value


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Budget:
    expansions: int | None = None
    queries: int | None = None

    def __post_init__(self):
        for value in (self.expansions, self.queries):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Budgets must be nonnegative integers or None")


@dataclass
class Result:
    best_state: Any = None
    best_value: float | None = None
    stop: str = "exhausted"
    expanded: int = 0
    generated: int = 0
    scored: int = 0
    queries: int = 0
    rejected: int = 0
    pruned: int = 0
    max_depth: int = 0
    peak_pending: int = 1


class Run:
    """One run's counters and terminal/budget rules, shared by all schedulers."""

    def __init__(self, problem, budget, event):
        self.problem, self.budget, self.event = problem, budget, event
        self.result = Result()
        self.stopped = budget.queries == 0
        if self.stopped:
            self.result.stop = "query_budget"

    def emit(self, kind, **fields):
        if self.event is not None:
            self.event({"kind": kind, **fields})

    def visit(self, state, depth):
        """Return children or None for terminal/rejected/stopped.

        Evaluate terminals on visitation, never on generation. An expansion cap
        stops at the next nonterminal requiring expansion; it does NOT skip that
        node to inspect later queued terminals. A query cap stops immediately.
        """
        p, r = self.problem, self.result
        r.max_depth = max(r.max_depth, depth)
        self.emit("visit", state=state, depth=depth)
        if not p.feasible(state):
            r.rejected += 1
            return None
        if p.terminal(state):
            value = finite(p.objective(state))
            r.queries += 1
            if r.best_value is None or value < r.best_value:
                r.best_state, r.best_value = state, value
            self.emit("terminal", state=state, value=value)
            if self.budget.queries is not None and r.queries >= self.budget.queries:
                r.stop, self.stopped = "query_budget", True
            return None
        if self.budget.expansions is not None and r.expanded >= self.budget.expansions:
            r.stop, self.stopped = "expansion_budget", True
            return None
        r.expanded += 1
        children = []
        for child in p.successors(state):
            r.generated += 1
            if p.feasible(child):
                children.append(child)
            else:
                r.rejected += 1
        return children

    def rank(self, children):
        ranked = [(finite(self.problem.heuristic(child)), child) for child in children]
        self.result.scored += len(ranked)
        # Key-only stable sorting never compares state objects to break ties.
        ranked.sort(key=lambda pair: pair[0])
        return ranked

    def pressure(self, count):
        # Queue records only, NOT bytes or total temporary/model memory.
        self.result.peak_pending = max(self.result.peak_pending, count)
