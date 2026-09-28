"""Release-specific, restricted streaming reader for the 5 GB legacy archive.

Python's ordinary pickle/torch loader retains the whole object graph. This reader
extracts each architecture at a time, then discards its bulky training histories.
Only explicitly allowed constructors can run. It rejects cross-record references
to discarded containers rather than silently supplying wrong data. Not a generic
untrusted-pickle sandbox: only use the checksum-verified official release.
"""

import argparse
import codecs
import hashlib
import json
import pickle
from array import array
from collections import OrderedDict
from pathlib import Path

import numpy as np

from nas_space import FILENAME, parse_architecture

EXPECTED_MD5 = "55e847143ce1f7c2d89b676f6b096897"


class CompactMemo:
    """Compact memo IDs plus interned immutable objects and one live record.

    Positive IDs identify persistent immutable values; negative IDs identify
    record-local containers. Released containers cannot subsequently be read.
    """

    def __init__(self):
        self.ids = array("i")
        self.values = [None]
        self.intern = {}
        self.live = {}

    def __setitem__(self, index, value):
        if index >= len(self.ids):
            self.ids.extend([0] * (index + 1 - len(self.ids)))
        immutable_tuple = isinstance(value, tuple) and all(
            isinstance(item, (str, bytes, int, float, type(None))) for item in value
        )
        if (
            immutable_tuple
            or isinstance(value, (str, bytes, int, float, type(None), np.dtype))
            or callable(value)
        ):
            key = (type(value), value)
            code = self.intern.get(key)
            if code is None:
                code = len(self.values)
                self.intern[key] = code
                self.values.append(value)
            self.ids[index] = code
        else:
            self.ids[index] = -index - 1
            self.live[index] = value

    def __getitem__(self, index):
        code = self.ids[index]
        if code > 0:
            return self.values[code]
        if code == 0 or index not in self.live:
            raise ValueError(f"Unsupported cross-record container reference at memo {index}")
        return self.live[index]

    def release_record(self):
        self.live.clear()


def summarize(record):
    """Extract only final CIFAR validation accuracy; never export test labels."""
    full = record["full"]
    state = parse_architecture(full["arch_str"])
    trials = []
    for (dataset, seed), result in full["all_results"].items():
        if dataset != "cifar10-valid":
            continue
        if result["epochs"] != 200:
            raise ValueError("Expected exactly 200 training epochs")
        accuracy = float(result["eval_acc1es"]["x-valid@199"])
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
            raise ValueError("Invalid recorded accuracy")
        trials.append({"seed": int(seed), "accuracy": accuracy})
    if not trials or len({t["seed"] for t in trials}) != len(trials):
        raise ValueError("Missing or duplicate CIFAR validation trials")
    trials.sort(key=lambda trial: trial["seed"])
    return {
        "index": int(full["arch_index"]),
        "state": list(state),
        "trials": trials,
        "error": 100 - sum(t["accuracy"] for t in trials) / len(trials),
    }


class StreamingReader(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, stream, callback):
        super().__init__(stream, encoding="latin1")
        self.memo = CompactMemo()
        self.callback = callback

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("_codecs", "encode"): codecs.encode,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported pickle constructor: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, pid):
        raise ValueError("Tensor storage is not supported by this metadata extractor")

    def load_setitems(self):
        super().load_setitems()
        target = self.stack[-1]
        if "full" in target and "less" in target:
            row = summarize(target)
            self.callback(row)
            target.clear()
            self.memo.release_record()

    dispatch[pickle.SETITEMS[0]] = load_setitems


def extract(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    print("Verifying official release checksum...", flush=True)
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    if checksum != EXPECTED_MD5:
        raise ValueError("Dataset checksum does not match the supported release")
    records = []

    def collect(row):
        records.append(row)
        if len(records) % 250 == 0:
            print(f"Extracted {len(records)}/15625 architectures", flush=True)

    with source.open("rb") as stream:
        # The first three records are torch's scalar/header metadata, not data.
        for _ in range(3):
            StreamingReader(stream, collect).load()
        StreamingReader(stream, collect).load()
    if len(records) != 15625 or {row["index"] for row in records} != set(range(15625)):
        raise ValueError("Incomplete or duplicate architecture records")
    if len({tuple(row["state"]) for row in records}) != 15625:
        raise ValueError("Duplicate architecture encodings")
    payload = {
        "schema": 1,
        "source_md5": checksum,
        "dataset": "cifar10-valid",
        "metric": "x-valid",
        "epochs": 200,
        "aggregation": "available-trial-mean",
        "records": sorted(records, key=lambda row: row["index"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False)
    print(f"Saved {len(records)} validation-only records to {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(FILENAME))
    parser.add_argument("--output", type=Path, default=Path("artifacts/measurements.json"))
    args = parser.parse_args()
    extract(args.dataset, args.output)


if __name__ == "__main__":
    main()
