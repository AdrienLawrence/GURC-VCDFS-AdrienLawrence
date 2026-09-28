"""Architecture-level splits and prefix examples; no torch dependency."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from nas_space import validate_prefix


def tokenize(prefix):
    """Six operation IDs; 5 marks every not-yet-assigned suffix position."""
    validate_prefix(prefix)
    return prefix + (5,) * (6 - len(prefix))


def load_measurements(path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if (
        payload.get("schema"),
        payload.get("dataset"),
        payload.get("metric"),
        payload.get("epochs"),
        payload.get("aggregation"),
    ) != (1, "cifar10-valid", "x-valid", 200, "available-trial-mean"):
        raise ValueError("Wrong measurement schema/objective")
    records = payload["records"]
    if len(records) != 15625 or {r["index"] for r in records} != set(range(15625)):
        raise ValueError("Expected exactly 15625 unique indexed architectures")
    states = set()
    for row in records:
        state = tuple(row["state"])
        validate_prefix(state)
        if len(state) != 6 or state in states:
            raise ValueError("Invalid or duplicate architecture")
        states.add(state)
        trials = row["trials"]
        if not trials or len({t["seed"] for t in trials}) != len(trials):
            raise ValueError("Missing/duplicate trial seeds")
        accuracy = [float(t["accuracy"]) for t in trials]
        if not all(np.isfinite(x) and 0 <= x <= 100 for x in accuracy):
            raise ValueError("Invalid trial accuracy")
        error = float(row["error"])
        if not np.isfinite(error) or abs(error - (100 - sum(accuracy) / len(accuracy))) > 1e-9:
            raise ValueError("Error label does not match recorded trial mean")
    return sorted(records, key=lambda r: r["index"]), hashlib.sha256(raw).hexdigest()


def split_indices(size, seed, train_size=1024, dev_size=256):
    if any(type(x) is not int or x < 1 for x in (size, train_size, dev_size)):
        raise ValueError("Split sizes must be positive integers")
    if train_size + dev_size >= size:
        raise ValueError("A nonempty held-out set is required")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return {
        "train": indices[:train_size],
        "dev": indices[train_size : train_size + dev_size],
        "test": indices[train_size + dev_size :],
    }


def select_records(records, indices):
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate selected architecture indices")
    return [records[i] for i in indices]


def examples(records):
    """Six examples per architecture. Repeated prefixes/outcomes are intentional.

    MSE's empirical optimum for a prefix is its mean observed error. Grouping
    identical prefixes is equivalent only if weighted by observation count.
    """
    if not records:
        raise ValueError("Cannot create an empty training set")
    x, y, depths, seen = [], [], [], set()
    for row in records:
        state = tuple(row["state"])
        validate_prefix(state)
        if len(state) != 6 or state in seen:
            raise ValueError("Examples require unique complete architectures")
        seen.add(state)
        error = float(row["error"])
        if not np.isfinite(error) or not 0 <= error <= 100:
            raise ValueError("Validation error must be in [0, 100]")
        for depth in range(1, 7):
            x.append(tokenize(state[:depth]))
            y.append(error)
            depths.append(depth)
    return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.float64), np.asarray(depths)
