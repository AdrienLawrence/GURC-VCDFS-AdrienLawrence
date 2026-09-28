"""NAS-Bench-201 metadata and six-decision construction space, standard library only."""

import io
import pickletools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Native string order: grouped by destination node.
EDGES = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))
OPERATIONS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
FILENAME = "NAS-Bench-201-v1_1-096897.pth"


def validate_prefix(prefix):
    if not isinstance(prefix, tuple) or len(prefix) > 6:
        raise ValueError("State must be a tuple with at most six operation IDs")
    if any(type(x) is not int or not 0 <= x < 5 for x in prefix):
        raise ValueError("Operation IDs must be integers in [0, 4]")


def parse_architecture(text):
    groups = text.split("+")
    if len(groups) != 3:
        raise ValueError("Expected three destination-node groups")
    result = []
    for destination, group in enumerate(groups, start=1):
        if not group.startswith("|") or not group.endswith("|"):
            raise ValueError("Each group must be enclosed by pipes")
        entries = group[1:-1].split("|")
        if len(entries) != destination:
            raise ValueError("Wrong number of incoming edges")
        for source, entry in enumerate(entries):
            operation, origin = entry.split("~")
            if origin != str(source) or operation not in OPERATIONS:
                raise ValueError("Invalid edge operation or source order")
            result.append(OPERATIONS.index(operation))
    return tuple(result)


def architecture_string(state):
    validate_prefix(state)
    if len(state) != 6:
        raise ValueError("Only complete architectures have benchmark strings")
    groups = []
    offset = 0
    for destination in range(1, 4):
        group = (
            "|"
            + "|".join(
                f"{OPERATIONS[state[offset + source]]}~{source}" for source in range(destination)
            )
            + "|"
        )
        groups.append(group)
        offset += destination
    return "+".join(groups)


def encode(prefix):
    """Six one-hot blocks of length six; index 5 denotes unassigned."""
    validate_prefix(prefix)
    values = prefix + (5,) * (6 - len(prefix))
    return tuple(float(category == value) for value in values for category in range(6))


def read_metadata(path):
    """Read only the leading meta_archs list, without executing pickle payloads.

    Deliberately specific to the downloaded legacy v1.1 file layout. This does
    not load measurement records or provide objective labels. It reads at most
    4 MiB and uses pickle opcode inspection rather than pickle.load/torch.load.
    """
    with Path(path).open("rb") as stream:
        data = io.BytesIO(stream.read(4 * 1024 * 1024))
    try:
        for _ in range(3):
            for _opcode in pickletools.genops(data):
                pass
        found = False
        architectures = []
        for op, value, _ in pickletools.genops(data):
            if op.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
                if not found:
                    if value != "meta_archs":
                        raise ValueError("Unexpected leading field; expected meta_archs")
                    found = True
                else:
                    parse_architecture(value)
                    architectures.append(value)
                    if len(architectures) == 15625:
                        if len(set(architectures)) != 15625:
                            raise ValueError("Duplicate architecture strings")
                        return architectures
    except (ValueError, IndexError, UnicodeError) as exc:
        raise ValueError("Unsupported or corrupt NAS-Bench-201 v1.1 metadata") from exc
    raise ValueError("Architecture metadata list is incomplete")


@dataclass
class ArchitectureTree:
    """Full tree by default; optionally a trie of explicit allowed architectures.

    predictor consumes the partial operation-ID tuple; evaluator consumes a full
    six-ID tuple. This adapter provides neither a trained model nor labels.
    """

    predictor: Callable
    evaluator: Callable
    allowed: tuple | None = None
    initial: tuple = ()

    def __post_init__(self):
        self._prefixes = None
        if self.initial != ():
            raise ValueError("Initial architecture must be empty")
        if self.allowed is not None:
            self._prefixes = {()}
            for state in self.allowed:
                validate_prefix(state)
                if len(state) != 6:
                    raise ValueError("Allowed architectures must be complete")
                self._prefixes.update(state[:depth] for depth in range(1, 7))

    def feasible(self, state):
        validate_prefix(state)
        return self._prefixes is None or state in self._prefixes

    def terminal(self, state):
        return len(state) == 6

    def successors(self, state):
        if len(state) < 6:
            for operation in range(5):
                child = state + (operation,)
                if self.feasible(child):
                    yield child

    def heuristic(self, state):
        return self.predictor(state)

    def objective(self, state):
        if not self.terminal(state):
            raise ValueError("Objective queries require complete architectures")
        return self.evaluator(state)
