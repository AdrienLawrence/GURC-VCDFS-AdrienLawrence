"""Canonical DARTS encoding; conversion follows official representations.py."""

from collections import namedtuple
from itertools import combinations

OPS = (
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
)
OP_TO_TOKEN = {operation: index for index, operation in enumerate(OPS)}
TOKEN_TO_OP = dict(enumerate(OPS))
PAIRS = tuple(combinations(range(5), 2))
PAIR_TO_TOKEN = {pair: index + 7 for index, pair in enumerate(PAIRS)}
TOKEN_TO_PAIR = {token: pair for pair, token in PAIR_TO_TOKEN.items()}
UNASSIGNED = 17
VOCABULARY_SIZE = 18
SEQUENCE_LENGTH = 22
ENCODING = "darts-22-decision-v1"
TOKEN_MAPPING = "darts-ops7-pairs10-v1"
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
BASE = "NetworkSelectorDatasetInfo:darts:"
DECISIONS = tuple(
    (cell, node, kind, slot)
    for cell in ("normal", "reduce")
    for node in range(2, 6)
    for kind, slot in (
        (("op", 0), ("op", 1)) if node == 2 else (("pair", None), ("op", 0), ("op", 1))
    )
)
ALLOWED_VALUES = tuple(
    tuple(range(7))
    if kind == "op"
    else tuple(PAIR_TO_TOKEN[pair] for pair in combinations(range(node), 2))
    for _, node, kind, _ in DECISIONS
)
BRANCHING_FACTORS = tuple(map(len, ALLOWED_VALUES))
DECISION_TYPES = tuple(0 if kind == "op" else 1 for _, _, kind, _ in DECISIONS)
CELL_TYPES = (0,) * 11 + (1,) * 11


def validate_prefix(state):
    if not isinstance(state, tuple) or len(state) > SEQUENCE_LENGTH:
        raise ValueError("DARTS state must be a tuple of at most 22 tokens")
    for depth, token in enumerate(state):
        if type(token) is not int or token not in ALLOWED_VALUES[depth]:
            raise ValueError(f"Illegal token {token!r} at DARTS position {depth}")


def allowed_tokens(depth):
    if type(depth) is not int or not 0 <= depth <= SEQUENCE_LENGTH:
        raise ValueError("DARTS depth must be an integer in [0, 22]")
    return () if depth == SEQUENCE_LENGTH else ALLOWED_VALUES[depth]


def branching_factor(depth):
    return len(allowed_tokens(depth))


def genotype_to_state(genotype):
    state = []
    for cell in ("normal", "reduce"):
        if list(getattr(genotype, cell + "_concat")) != [2, 3, 4, 5]:
            raise ValueError("DARTS concat must be [2, 3, 4, 5]")
        edges = list(getattr(genotype, cell))
        if len(edges) != 8:
            raise ValueError("Every DARTS cell must contain eight selected edges")
        for offset, node in enumerate(range(2, 6)):
            group = edges[2 * offset : 2 * offset + 2]
            if any(
                not isinstance(edge, (tuple, list))
                or len(edge) != 2
                or edge[0] not in OP_TO_TOKEN
                or type(edge[1]) is not int
                or not 0 <= edge[1] < node
                for edge in group
            ):
                raise ValueError(f"Invalid selected edge for {cell} node {node}")
            group = sorted(group, key=lambda edge: edge[1])
            pair = tuple(source for _, source in group)
            if len(set(pair)) != 2:
                raise ValueError("Selected predecessors must be distinct")
            if node != 2:
                state.append(PAIR_TO_TOKEN[pair])
            state.extend(OP_TO_TOKEN[operation] for operation, _ in group)
    result = tuple(state)
    validate_prefix(result)
    return result


def state_to_genotype(state):
    validate_prefix(state)
    if len(state) != SEQUENCE_LENGTH:
        raise ValueError("A terminal DARTS genotype requires all 22 decisions")
    cells, position = [], 0
    for _ in range(2):
        edges = []
        for node in range(2, 6):
            pair = (0, 1)
            if node != 2:
                pair = TOKEN_TO_PAIR[state[position]]
                position += 1
            for source in pair:
                edges.append((TOKEN_TO_OP[state[position]], source))
                position += 1
        cells.append(edges)
    return Genotype(cells[0], [2, 3, 4, 5], cells[1], [2, 3, 4, 5])


def genotype_json(state):
    """Return a JSON-safe genotype without tuple-valued edge records."""
    genotype = state_to_genotype(state)
    return {
        "normal": [list(edge) for edge in genotype.normal],
        "normal_concat": list(genotype.normal_concat),
        "reduce": [list(edge) for edge in genotype.reduce],
        "reduce_concat": list(genotype.reduce_concat),
    }


def config_to_state(config):
    if not isinstance(config, dict):
        config = dict(config)
    cells = []
    for cell in ("normal", "reduce"):
        edges, start = [], 0
        for node in range(2, 6):
            if node == 2:
                pair = (0, 1)
            else:
                value = config.get(f"{BASE}inputs_node_{cell}_{node}")
                if not isinstance(value, str):
                    raise ValueError(f"Missing predecessor pair for {cell} node {node}")
                try:
                    pair = tuple(int(source) for source in value.split("_"))
                except ValueError as error:
                    raise ValueError("Malformed ConfigSpace predecessor pair") from error
                if len(pair) != 2 or len(set(pair)) != 2 or any(not 0 <= x < node for x in pair):
                    raise ValueError("Invalid ConfigSpace predecessor pair")
                pair = tuple(sorted(pair))
            for source in pair:
                operation = config.get(f"{BASE}edge_{cell}_{start + source}")
                if operation not in OP_TO_TOKEN:
                    raise ValueError(
                        f"Missing/illegal operation on {cell} node {node} source {source}"
                    )
                edges.append((operation, source))
            start += node
        cells.append(edges)
    return genotype_to_state(Genotype(cells[0], [2, 3, 4, 5], cells[1], [2, 3, 4, 5]))


def state_to_config(state):
    genotype = state_to_genotype(state)
    config = {}
    for cell in ("normal", "reduce"):
        edges, start = getattr(genotype, cell), 0
        for offset, node in enumerate(range(2, 6)):
            group = edges[2 * offset : 2 * offset + 2]
            for operation, source in group:
                config[f"{BASE}edge_{cell}_{start + source}"] = operation
            if node != 2:
                config[f"{BASE}inputs_node_{cell}_{node}"] = "_".join(
                    str(source) for _, source in group
                )
            start += node
    return config


def random_complete_state(rng):
    return tuple(rng.choice(values) for values in ALLOWED_VALUES)


def random_completion(prefix, rng):
    validate_prefix(prefix)
    return prefix + tuple(rng.choice(values) for values in ALLOWED_VALUES[len(prefix) :])
