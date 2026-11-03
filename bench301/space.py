"""DARTS construction tree implementing the shared Problem interface."""

from dataclasses import dataclass
from typing import Callable

from .representation import SEQUENCE_LENGTH, allowed_tokens, validate_prefix


@dataclass
class ArchitectureTree:
    predictor: Callable
    evaluator: Callable
    initial: tuple = ()
    batch_scores: bool = False

    def __post_init__(self):
        if self.initial != ():
            raise ValueError("Initial DARTS architecture must be empty")

    def feasible(self, state):
        validate_prefix(state)
        return True

    def terminal(self, state):
        validate_prefix(state)
        return len(state) == SEQUENCE_LENGTH

    def successors(self, state):
        validate_prefix(state)
        for token in allowed_tokens(len(state)):
            yield state + (token,)

    def heuristic(self, state):
        validate_prefix(state)
        return self.predictor(state)

    def heuristics(self, states):
        """Score siblings together when the frozen predictor supports batching."""
        for state in states:
            validate_prefix(state)
        if hasattr(self.predictor, "predict"):
            return self.predictor.predict(states)
        return [self.predictor(state) for state in states]

    def objective(self, state):
        if not self.terminal(state):
            raise ValueError("NB301 objective queries require a complete 22-decision state")
        return self.evaluator(state)
