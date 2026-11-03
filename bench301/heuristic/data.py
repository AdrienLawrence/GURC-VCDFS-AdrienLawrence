"""Architecture-level splitting and a lazy 22-prefix-per-architecture dataset."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bench301.representation import ENCODING, SEQUENCE_LENGTH, UNASSIGNED, validate_prefix


def tokenize(prefix):
    validate_prefix(prefix)
    return prefix + (UNASSIGNED,) * (SEQUENCE_LENGTH - len(prefix))


def validate_records(records):
    if not records:
        raise ValueError("Measured architectures must be nonempty")
    seen = set()
    for row in records:
        state = tuple(row["state"])
        validate_prefix(state)
        if len(state) != SEQUENCE_LENGTH or state in seen:
            raise ValueError("Measurements require unique complete canonical architectures")
        seen.add(state)
        error = row["error"]
        if isinstance(error, bool) or not np.isfinite(error) or not 0 <= error <= 100:
            raise ValueError("Measured validation error must be finite and in [0, 100]")


def load_measurements(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing real NB301 measurements: {path}; run bench301.heuristic.extract"
        )
    raw = path.read_bytes()
    payload = json.loads(raw)
    expected = {
        "schema": 1,
        "benchmark": "nasbench301",
        "encoding": ENCODING,
        "dataset": "cifar10",
        "metric": "info[0].val_accuracy",
        "units": "percentage_points",
        "aggregation": "available-observation-mean",
        "label_source": "real-measured-training-records",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Wrong NB301 measurement schema/objective/provenance")
    if not payload.get("sources") or not payload.get("source_version"):
        raise ValueError("Missing source provenance")
    for source in payload["sources"]:
        if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
            raise ValueError("Missing source checksum")
    records = payload["records"]
    validate_records(records)
    if payload.get("record_count") != len(records):
        raise ValueError("Measurement count mismatch")
    if [row["index"] for row in records] != list(range(len(records))):
        raise ValueError("Measurement indices must be consecutive")
    for row in records:
        trials = row.get("trials", [])
        if not trials:
            raise ValueError("Missing measured observations")
        accuracy = [trial["validation_accuracy"] for trial in trials]
        if any(isinstance(x, bool) or not np.isfinite(x) or not 0 <= x <= 100 for x in accuracy):
            raise ValueError("Invalid measured accuracy")
        if abs(row["error"] - (100 - sum(accuracy) / len(accuracy))) > 1e-9:
            raise ValueError("Error differs from measured observation mean")
    return records, hashlib.sha256(raw).hexdigest()


def split_indices(size, seed, train_size=None, dev_size=None):
    if type(size) is not int or size < 3:
        raise ValueError("At least three architectures are required")
    train_size = max(1, int(size * 0.8)) if train_size is None else train_size
    dev_size = max(1, int(size * 0.1)) if dev_size is None else dev_size
    if any(type(x) is not int or x < 1 for x in (train_size, dev_size)):
        raise ValueError("Split sizes must be positive integers")
    if train_size + dev_size >= size:
        raise ValueError("A nonempty held-out architecture partition is required")
    order = list(range(size))
    random.Random(seed).shuffle(order)
    return {
        "train": order[:train_size],
        "dev": order[train_size : train_size + dev_size],
        "test": order[train_size + dev_size :],
    }


def select_records(records, indices):
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate selected architecture indices")
    return [records[index] for index in indices]


class PrefixDataset(Dataset):
    """Store complete architectures once; construct prefixes only when requested."""

    def __init__(self, records, mean=0.0, scale=1.0):
        validate_records(records)
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid target normalization")
        self.states = np.asarray([row["state"] for row in records], dtype=np.int64)
        self.errors = np.asarray([row["error"] for row in records], dtype=np.float64)
        self.mean, self.scale = mean, scale

    def __len__(self):
        return len(self.states) * SEQUENCE_LENGTH

    def __getitem__(self, index):
        architecture, offset = divmod(index, SEQUENCE_LENGTH)
        depth = offset + 1
        tokens = self.states[architecture].copy()
        tokens[depth:] = UNASSIGNED
        target = (self.errors[architecture] - self.mean) / self.scale
        return torch.from_numpy(tokens), torch.tensor(target, dtype=torch.float32)
