"""Prespecified search controls for NAS-Bench-301."""

from search_registry import AlgorithmSpec


def parameter_specs(preset):
    if preset == "v-family":
        specifications = []
        proportional = ((10, 30), (20, 60), (40, 100))
        absolute_depth = ((1, 2), (1, 3), (2, 4))
        absolute_breadth = ((1, 1), (1, 2), (2, 3))
        for algorithm in ("vcdfs", "vcbfs", "vcbfsn"):
            for lower, upper in proportional:
                specifications.append(
                    AlgorithmSpec(
                        f"{algorithm.upper()}-{lower}-{upper}pct",
                        algorithm,
                        {"pmin": lower, "pmax": upper},
                    )
                )
        for algorithm in ("vcdfsa", "vcbfsa", "vcbfsna"):
            bounds = absolute_depth if algorithm == "vcdfsa" else absolute_breadth
            for lower, upper in bounds:
                specifications.append(
                    AlgorithmSpec(
                        f"{algorithm[:-1].upper()}-A-{lower}-{upper}",
                        algorithm,
                        {"kmin": lower, "kmax": upper},
                    )
                )
        for lower, upper in proportional[:2]:
            specifications.append(
                AlgorithmSpec(
                    f"VCBFSBeam-{lower}-{upper}pct",
                    "vcbfs_beam",
                    {"pmin": lower, "pmax": upper},
                )
            )
        return specifications
    if preset == "baselines":
        return [
            AlgorithmSpec("DFS", "dfs", {}),
            AlgorithmSpec("DFS-H", "heuristic_dfs", {}),
            AlgorithmSpec("BFS", "bfs", {}),
            AlgorithmSpec("GBFS", "gbfs", {}),
            AlgorithmSpec("AStar-depth+h", "astar", {}),
            AlgorithmSpec("MCTS-UCT", "mcts", {"seed": 0}),
            AlgorithmSpec("CDFS-2", "cdfs", {"k": 2}),
            AlgorithmSpec("CDFS-4", "cdfs", {"k": 4}),
            AlgorithmSpec("Beam-2", "beam", {"width": 2}),
            AlgorithmSpec("Beam-4", "beam", {"width": 4}),
            AlgorithmSpec("Beam-8", "beam", {"width": 8}),
            AlgorithmSpec(
                "DynamicBeam-Entropy-2-8",
                "dynamic_beam",
                {
                    "policy": "entropy",
                    "minimum_width": 2,
                    "maximum_width": 8,
                    "slope": 4.0,
                    "intercept": 0.0,
                    "temperature": 1.0,
                },
            ),
            AlgorithmSpec(
                "DynamicBeam-StdDev-2-8",
                "dynamic_beam",
                {
                    "policy": "standard_deviation",
                    "minimum_width": 2,
                    "maximum_width": 8,
                    "slope": -100.0,
                    "intercept": 8.0,
                    "top_k": 5,
                    "temperature": 1.0,
                },
            ),
        ]
    raise ValueError("preset must be v-family or baselines")
