"""Seeded Monte Carlo Tree Search using the UCT selection rule."""

import math
import random
from dataclasses import dataclass, field

from .common import Budget, Run, finite

DEFAULT_EXPLORATION = math.sqrt(2)


@dataclass
class _Node:
    visits: int = 0
    total_reward: float = 0.0
    children: list = field(default_factory=list)
    expanded: bool = False
    exhausted: bool = False


def mcts(problem, exploration=DEFAULT_EXPLORATION, seed=0, *, budget=Budget(), event=None):
    """Evaluate unique terminal rollouts until the budget/tree is exhausted.

    Lower objective values become higher rewards by negation. Generated children
    are shuffled by a private seeded generator, making every run reproducible.
    """
    exploration = finite(exploration)
    if exploration < 0:
        raise ValueError("exploration must be nonnegative")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    terminal_value = None

    def observe(message):
        nonlocal terminal_value
        if message["kind"] == "terminal":
            terminal_value = message["value"]
        if event is not None:
            event(message)

    run = Run(problem, budget, observe)
    rng = random.Random(seed)
    nodes = {problem.initial: _Node()}

    def choose(node):
        available = [state for state in node.children if not nodes[state].exhausted]
        unvisited = [state for state in available if nodes[state].visits == 0]
        if unvisited:
            return unvisited[0]
        parent_log = math.log(max(1, node.visits))
        return max(
            available,
            key=lambda state: (
                nodes[state].total_reward / nodes[state].visits
                + exploration * math.sqrt(parent_log / nodes[state].visits)
            ),
        )

    while not nodes[problem.initial].exhausted and not run.stopped:
        state, depth = problem.initial, 0
        path = [state]
        terminal_value = None
        while not run.stopped:
            node = nodes[state]
            if not node.expanded:
                children = run.visit(state, depth)
                node.expanded = True
                if children is None:
                    node.exhausted = problem.terminal(state)
                    break
                node.children = list(children)
                rng.shuffle(node.children)
                nodes.update((child, _Node()) for child in node.children)
                if not node.children:
                    node.exhausted = True
                    break
            available = [child for child in node.children if not nodes[child].exhausted]
            if not available:
                node.exhausted = True
                break
            state = choose(node)
            depth += 1
            path.append(state)
        if terminal_value is not None:
            reward = -terminal_value
            for visited in path:
                nodes[visited].visits += 1
                nodes[visited].total_reward += reward
            run.emit("mcts_rollout", state=path[-1], depth=depth, reward=reward)
        for visited in reversed(path):
            node = nodes[visited]
            if node.expanded and node.children:
                node.exhausted = all(nodes[child].exhausted for child in node.children)
        run.pressure(sum(not node.exhausted for node in nodes.values()))
    return run.result
