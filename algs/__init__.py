"""Public algorithms plus a small dispatcher for existing experiment settings."""

from .beam import beam
from .bfs import bfs
from .cdfs import cdfs
from .common import Budget, Problem, Result
from .dfs import dfs, heuristic_dfs
from .dynamic_beam import DynamicBeamConfig, dynamic_beam
from .gbfs import gbfs
from .vcbfs import VCBFSConfig, breadth_confidence, vcbfs, vcbfsa
from .vcdfs import Retention, confidence, vcdfs, vcdfsa

__all__ = [
    "Budget",
    "Problem",
    "Result",
    "Retention",
    "DynamicBeamConfig",
    "VCBFSConfig",
    "confidence",
    "search",
    "dfs",
    "heuristic_dfs",
    "bfs",
    "beam",
    "dynamic_beam",
    "gbfs",
    "cdfs",
    "vcdfs",
    "vcdfsa",
    "breadth_confidence",
    "vcbfs",
    "vcbfsa",
]


def search(problem, policy=None, *, schedule="dfs", beam_width=16, budget=Budget(), event=None):
    if schedule != "dfs" and policy is not None:
        raise ValueError("Local retention is only valid with DFS")
    if schedule == "dfs":
        return dfs(
            problem, selector=None if policy is None else policy.width, budget=budget, event=event
        )
    if schedule == "bfs":
        return bfs(problem, budget=budget, event=event)
    if schedule == "beam":
        return beam(problem, width=beam_width, budget=budget, event=event)
    raise ValueError("Unknown schedule")
