"""Event-derived pruning and anytime metrics without changing search decisions."""

import time


def anytime_mean_regret(improvements, oracle, horizon):
    """Mean best-so-far regret over query positions 1..horizon.

    After an exhausted search, its final incumbent is carried forward through
    the requested horizon. This makes curves comparable at the same budget.
    """
    if horizon < 1 or not improvements:
        raise ValueError("Anytime regret needs a positive horizon and a terminal result")
    ordered = sorted(improvements, key=lambda row: row["query"])
    best = None
    position = 0
    total = 0.0
    for query in range(1, horizon + 1):
        while position < len(ordered) and ordered[position]["query"] <= query:
            value = float(ordered[position]["value"])
            best = value if best is None else min(best, value)
            position += 1
        if best is None:
            raise ValueError("The first terminal must occur at query one")
        total += best - oracle
    return total / horizon


class BehaviorObserver:
    """Track which global-optimal prefixes survive and when incumbents improve."""

    def __init__(self, optimal_states, oracle, terminal_depth=6):
        self.optimal_states = {tuple(state) for state in optimal_states}
        if not self.optimal_states:
            raise ValueError("At least one optimal state is required")
        self.surviving = set(self.optimal_states)
        self.oracle = float(oracle)
        if type(terminal_depth) is not int or terminal_depth < 1:
            raise ValueError("terminal_depth must be a positive integer")
        self.terminal_depth = terminal_depth
        self.started = time.perf_counter()
        self.expansions = 0
        self.scored = 0
        self.queries = 0
        self.prune_decisions = 0
        self.optimal_prune_events = []
        self.improvements = []
        self._best = None

    def __call__(self, event):
        kind = event["kind"]
        if kind == "visit" and len(event["state"]) < self.terminal_depth:
            self.expansions += 1
        elif kind == "terminal":
            self._terminal(event)
        elif kind == "rank":
            self.scored += int(event["count"])
        elif kind == "retain":
            self._retention_group(
                event["state"], event.get("retained", ()), event.get("pruned", ())
            )
        elif kind == "vcbfs":
            for group in event["groups"]:
                self._retention_group(group["parent"], group["retained"], group["pruned"])
        elif kind in {"beam", "dynamic_beam", "vcbfs_beam"}:
            self._global_retention(event.get("retained", ()), event.get("pruned", ()))

    def _terminal(self, event):
        self.queries += 1
        value = float(event["value"])
        if self._best is None or value < self._best:
            self._best = value
            self.improvements.append(
                {
                    "query": self.queries,
                    "expansions": self.expansions,
                    "scored": self.scored,
                    "elapsed_seconds": time.perf_counter() - self.started,
                    "value": value,
                    "regret_pp": value - self.oracle,
                    "state": list(event["state"]),
                }
            )

    def _retention_group(self, parent, retained, pruned):
        parent = tuple(parent)
        retained_states = {tuple(item["state"]) for item in retained}
        pruned_states = {tuple(item["state"]) for item in pruned}
        if pruned_states:
            self.prune_decisions += 1
        newly_pruned = {
            optimum
            for optimum in self.surviving
            if optimum[: len(parent)] == parent
            and optimum[: len(parent) + 1] in pruned_states
            and optimum[: len(parent) + 1] not in retained_states
        }
        if newly_pruned:
            self.surviving.difference_update(newly_pruned)
            self.optimal_prune_events.append(
                {
                    "parent": list(parent),
                    "depth": len(parent) + 1,
                    "query": self.queries,
                    "expansions": self.expansions,
                    "scored": self.scored,
                    "optimal_paths_removed": len(newly_pruned),
                    "optimal_paths_remaining": len(self.surviving),
                }
            )

    def _global_retention(self, retained, pruned):
        retained_states = {tuple(item["state"]) for item in retained}
        pruned_states = {tuple(item["state"]) for item in pruned}
        if not pruned_states:
            return
        self.prune_decisions += 1
        newly_pruned = {
            optimum
            for optimum in self.surviving
            if any(optimum[: len(state)] == state for state in pruned_states)
            and not any(optimum[: len(state)] == state for state in retained_states)
        }
        if newly_pruned:
            self.surviving.difference_update(newly_pruned)
            self.optimal_prune_events.append(
                {
                    "parent": None,
                    "depth": len(next(iter(pruned_states))),
                    "query": self.queries,
                    "expansions": self.expansions,
                    "scored": self.scored,
                    "optimal_paths_removed": len(newly_pruned),
                    "optimal_paths_remaining": len(self.surviving),
                }
            )

    def metrics(self, horizon):
        if not self.improvements:
            raise RuntimeError("Behavior run produced no terminal solution")
        final = self.improvements[-1]
        optimum_query = next(
            (row["query"] for row in self.improvements if row["regret_pp"] <= 1e-12), None
        )
        return {
            "optimal_architecture_count": len(self.optimal_states),
            "optimal_paths_surviving": len(self.surviving),
            "optimal_survival_fraction": len(self.surviving) / len(self.optimal_states),
            "any_optimum_pruned": int(len(self.surviving) < len(self.optimal_states)),
            "all_optima_pruned": int(not self.surviving),
            "optimum_found": int(optimum_query is not None),
            "query_of_optimum": optimum_query,
            "query_of_final_best": final["query"],
            "expansions_to_final_best": final["expansions"],
            "scores_to_final_best": final["scored"],
            "seconds_to_final_best": final["elapsed_seconds"],
            "anytime_mean_regret_pp": anytime_mean_regret(self.improvements, self.oracle, horizon),
            "prune_decisions": self.prune_decisions,
            "optimal_prune_events": self.optimal_prune_events,
            "improvement_trace": self.improvements,
        }
