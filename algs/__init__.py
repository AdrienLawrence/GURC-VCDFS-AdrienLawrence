"""Public algorithms plus a small dispatcher for existing experiment settings."""

from .astar import astar
from .beam import beam
from .bfs import bfs
from .cdfs import cdfs
from .common import Budget, Problem, Result
from .dfs import dfs, heuristic_dfs
from .dynamic_beam import DynamicBeamConfig, dynamic_beam
from .gbfs import gbfs
from .mcts import mcts
from .vcbfs import VCBFSConfig, breadth_confidence, vcbfs, vcbfsa
from .vcbfs_beam import vcbfs_beam, vcbfs_beama
from .vcbfsn import vcbfsn, vcbfsna
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
    "astar",
    "dynamic_beam",
    "gbfs",
    "mcts",
    "cdfs",
    "vcdfs",
    "vcdfsa",
    "breadth_confidence",
    "vcbfs",
    "vcbfsa",
    "vcbfs_beam",
    "vcbfs_beama",
    "vcbfsn",
    "vcbfsna",
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
