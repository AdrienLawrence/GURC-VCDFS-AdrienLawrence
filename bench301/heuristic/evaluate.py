"""Held-out measured prefix diagnostics; optional surrogate alignment is separate."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench301.representation import SEQUENCE_LENGTH, random_completion
from schemas import load_payload

from .data import load_measurements, select_records
from .model import FrozenPredictor, checkpoint_metadata


def metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if (
        not len(actual)
        or predicted.shape != actual.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("Invalid evaluation arrays")

    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]

    first, second = ranks(predicted), ranks(actual)
    spearman = (
        None if first.std() == 0 or second.std() == 0 else float(np.corrcoef(first, second)[0, 1])
    )
    return {
        "n": len(actual),
        "mae_pp": float(np.abs(predicted - actual).mean()),
        "rmse_pp": float(np.sqrt(((predicted - actual) ** 2).mean())),
        "spearman": spearman,
    }


def evaluate(predictor, records, batch_size=512):
    if not records or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation requires records and a positive batch size")
    actual = np.asarray([row["error"] for row in records], dtype=float)
    report = {"scope": "held-out real measured validation outcomes", "by_depth": {}}
    terminal_predictions = None
    for depth in range(1, SEQUENCE_LENGTH + 1):
        predictions = np.concatenate(
            [
                predictor.predict(
                    [tuple(row["state"][:depth]) for row in records[start : start + batch_size]]
                )
                for start in range(0, len(records), batch_size)
            ]
        )
        report["by_depth"][str(depth)] = metrics(predictions, actual)
        if depth == SEQUENCE_LENGTH:
            terminal_predictions = predictions
    cutoff = float(np.quantile(actual, 0.1))
    selected = actual <= cutoff
    report["best_decile_terminal"] = metrics(terminal_predictions[selected], actual[selected])
    report["best_decile_cutoff_pp"] = cutoff
    return report


def prefix_mean_diagnostics(predictor, records, batch_size=512):
    """Empirical means from held-out observations; not uniform full-space means."""
    groups = defaultdict(list)
    for row in records:
        for depth in range(1, SEQUENCE_LENGTH + 1):
            groups[tuple(row["state"][:depth])].append(row["error"])
    report = {}
    for depth in range(1, SEQUENCE_LENGTH + 1):
        prefixes = sorted(prefix for prefix in groups if len(prefix) == depth)
        predictions = np.concatenate(
            [
                predictor.predict(prefixes[start : start + batch_size])
                for start in range(0, len(prefixes), batch_size)
            ]
        )
        report[str(depth)] = metrics(predictions, [np.mean(groups[prefix]) for prefix in prefixes])
    return {
        "scope": "held-out empirical descendant means; sparse prefixes may have one observation",
        "by_depth": report,
    }


def surrogate_alignment(predictor, oracle, prefixes, completions=32, seed=0):
    """Optional uniform-completion diagnostic; never a training-label producer."""
    if type(completions) is not int or completions < 1 or not prefixes:
        raise ValueError("Alignment needs prefixes and positive completion count")
    rng = random.Random(seed)
    rows = []
    for prefix in prefixes:
        outcomes = [oracle(random_completion(tuple(prefix), rng)) for _ in range(completions)]
        rows.append(
            {
                "prefix": list(prefix),
                "prediction": predictor(tuple(prefix)),
                "completion_mean_error": float(np.mean(outcomes)),
                "completions": completions,
            }
        )
    return {
        "scope": "optional deterministic surrogate Monte Carlo alignment; not held-out evaluation",
        "seed": seed,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("bench301/artifacts/measurements.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    records, checksum = load_measurements(args.data)
    metadata = checkpoint_metadata(args.checkpoint / "model.json")
    if metadata["data_sha256"] != checksum:
        raise ValueError("Checkpoint/measurement checksum mismatch")
    split = load_payload(args.checkpoint / "split.json", "data_split")
    report = evaluate(
        FrozenPredictor.load(args.checkpoint / "model.json"), select_records(records, split["test"])
    )
    import json

    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
