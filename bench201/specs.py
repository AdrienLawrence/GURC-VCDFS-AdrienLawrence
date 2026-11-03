"""Unchanged NAS-Bench-201 default algorithm settings."""

from search_registry import RUNNERS, AlgorithmSpec, execute
from search_registry import load_specs as _load_specs

DEFAULT_SPECS = (
    AlgorithmSpec("DFS", "dfs", {}),
    AlgorithmSpec("BFS", "bfs", {}),
    AlgorithmSpec("DFS-H", "heuristic_dfs", {}),
    AlgorithmSpec("GBFS", "gbfs", {}),
    AlgorithmSpec("AStar", "astar", {}),
    AlgorithmSpec("MCTS-UCT", "mcts", {"exploration": 1.4142135623730951, "seed": 0}),
    *(AlgorithmSpec(f"CDFS-{width}", "cdfs", {"k": width}) for width in range(1, 6)),
    AlgorithmSpec("Beam-4", "beam", {"width": 4}),
    AlgorithmSpec("Beam-16", "beam", {"width": 16}),
    AlgorithmSpec(
        "DynamicBeam-Entropy-4-16",
        "dynamic_beam",
        {
            "policy": "entropy",
            "minimum_width": 4,
            "maximum_width": 16,
            "slope": 4.0,
            "intercept": 0.0,
            "temperature": 1.0,
        },
    ),
    AlgorithmSpec(
        "DynamicBeam-StdDev-4-16",
        "dynamic_beam",
        {
            "policy": "standard_deviation",
            "minimum_width": 4,
            "maximum_width": 16,
            "slope": -100.0,
            "intercept": 16.0,
            "top_k": 5,
            "temperature": 1.0,
        },
    ),
    AlgorithmSpec("VCDFS-20-40pct", "vcdfs", {"pmin": 20, "pmax": 40}),
    AlgorithmSpec("VCDFS-20-80pct", "vcdfs", {"pmin": 20, "pmax": 80}),
    AlgorithmSpec("VCDFS-20-100pct", "vcdfs", {"pmin": 20, "pmax": 100}),
    AlgorithmSpec("VCDFS-A-1-2", "vcdfsa", {"kmin": 1, "kmax": 2}),
    AlgorithmSpec("VCDFS-A-1-3", "vcdfsa", {"kmin": 1, "kmax": 3}),
    AlgorithmSpec("VCDFS-A-2-5", "vcdfsa", {"kmin": 2, "kmax": 5}),
    AlgorithmSpec("VCBFS-20-40pct", "vcbfs", {"pmin": 20, "pmax": 40}),
    AlgorithmSpec("VCBFS-20-60pct", "vcbfs", {"pmin": 20, "pmax": 60}),
    AlgorithmSpec("VCBFS-20-80pct", "vcbfs", {"pmin": 20, "pmax": 80}),
    AlgorithmSpec("VCBFSBeam-20-40pct", "vcbfs_beam", {"pmin": 20, "pmax": 40}),
    AlgorithmSpec("VCBFSBeam-20-80pct", "vcbfs_beam", {"pmin": 20, "pmax": 80}),
)


def load_specs(path=None):
    """Preserve NB201 defaults; custom files use the shared registry."""
    return list(DEFAULT_SPECS) if path is None else _load_specs(path)


__all__ = ["AlgorithmSpec", "DEFAULT_SPECS", "RUNNERS", "execute", "load_specs"]
