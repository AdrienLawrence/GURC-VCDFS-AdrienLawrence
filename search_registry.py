"""Benchmark-neutral algorithm specifications and dispatch."""

import json
from dataclasses import dataclass
from pathlib import Path

from algs import (
    DynamicBeamConfig,
    astar,
    beam,
    bfs,
    cdfs,
    dfs,
    dynamic_beam,
    gbfs,
    heuristic_dfs,
    mcts,
    vcbfs,
    vcbfs_beam,
    vcbfsa,
    vcbfsn,
    vcbfsna,
    vcdfs,
    vcdfsa,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    algorithm: str
    parameters: dict


RUNNERS = {
    "dfs": dfs,
    "bfs": bfs,
    "heuristic_dfs": heuristic_dfs,
    "gbfs": gbfs,
    "astar": astar,
    "mcts": mcts,
    "cdfs": cdfs,
    "beam": beam,
    "vcdfs": vcdfs,
    "vcdfsa": vcdfsa,
    "vcbfs": vcbfs,
    "vcbfsa": vcbfsa,
    "vcbfsn": vcbfsn,
    "vcbfsna": vcbfsna,
    "vcbfs_beam": vcbfs_beam,
}


def load_specs(path):
    """Load an optional user-defined algorithm list or return the documented defaults."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Algorithm specification must be a nonempty JSON list")
    specs = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"name", "algorithm", "parameters"}:
            raise ValueError("Each specification needs name, algorithm, and parameters")
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError("Algorithm name must be nonempty")
        if not isinstance(row["parameters"], dict):
            raise ValueError("Algorithm parameters must be an object")
        if row["algorithm"] not in RUNNERS and row["algorithm"] != "dynamic_beam":
            raise ValueError(f"Unknown algorithm: {row['algorithm']}")
        specs.append(AlgorithmSpec(row["name"], row["algorithm"], row["parameters"]))
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Algorithm names must be unique")
    return specs


def execute(spec, problem, budget, event=None):
    """Dispatch one specification without hiding algorithm-specific implementations."""
    if spec.algorithm == "dynamic_beam":
        return dynamic_beam(
            problem,
            DynamicBeamConfig(**spec.parameters),
            budget=budget,
            event=event,
        )
    return RUNNERS[spec.algorithm](problem, budget=budget, event=event, **spec.parameters)
