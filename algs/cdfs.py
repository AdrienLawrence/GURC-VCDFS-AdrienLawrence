"""Cluster DFS retains the k best children of EACH expanded parent."""

from .common import positive_integer
from .dfs import dfs


def cdfs(problem, k=2, **kwargs):
    positive_integer(k, "k")
    return dfs(problem, selector=lambda scores: (min(k, len(scores)), None), **kwargs)
