"""Conservative initial NB301 transfer controls; not a tuned publication grid."""

from search_registry import AlgorithmSpec
from search_registry import load_specs as _load_specs

# The existing VCDFS-A rule interpolates global bounds THEN clips to local b.
# Phase-3 before-interpolation clipping is intentionally not applied.
DEFAULT_SPECS = (
    AlgorithmSpec("VCDFS-A-2-10", "vcdfsa", {"kmin": 2, "kmax": 10}),
    AlgorithmSpec("CDFS-2", "cdfs", {"k": 2}),
    AlgorithmSpec("CDFS-3", "cdfs", {"k": 3}),
    AlgorithmSpec("DFS-H", "heuristic_dfs", {}),
    AlgorithmSpec("Beam-4", "beam", {"width": 4}),
)


def load_specs(path=None):
    return list(DEFAULT_SPECS) if path is None else _load_specs(path)
