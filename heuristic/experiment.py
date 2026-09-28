"""Reproducible heuristic training/evaluation ONLY. Never runs search algorithms."""

import argparse
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import load_measurements, select_records, split_indices
from .evaluate import Baselines, baseline_evaluation, evaluate, oracle_diagnostics
from .train import TrainingConfig, fit


def save_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def source_digest():
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in [
        root / "nas_space.py",
        *sorted((root / "algs").glob("*.py")),
        *sorted((root / "heuristic").glob("*.py")),
    ]:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/measurements.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[101, 203, 307, 409, 503])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if any(len(values) != len(set(values)) for values in (args.split_seeds, args.model_seeds)):
        parser.error("Seeds must be unique")
    config = TrainingConfig(epochs=args.epochs)
    args.output.mkdir(parents=True, exist_ok=False)
    records, checksum = load_measurements(args.data)
    manifest = {
        "scope": "FROZEN TRANSFORMER HEURISTIC ONLY; no algorithm experiments",
        "data_sha256": checksum,
        "source_sha256": source_digest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "split_seeds": args.split_seeds,
        "model_seeds": args.model_seeds,
        "training_config": asdict(config),
        "trial_count_histogram": dict(Counter(len(r["trials"]) for r in records)),
        "trial_seed_sets": dict(Counter(str([t["seed"] for t in r["trials"]]) for r in records)),
    }
    save_json(args.output / "manifest.json", manifest)
    summary = []
    for split_seed in args.split_seeds:
        split = split_indices(len(records), split_seed)
        train, dev, test = [select_records(records, split[key]) for key in ("train", "dev", "test")]
        baseline = baseline_evaluation(Baselines(train), test)
        for model_seed in args.model_seeds:
            name = f"split-{split_seed}-model-{model_seed}"
            directory = args.output / name
            directory.mkdir()
            save_json(directory / "split.json", split)
            print(f"Training {name}", flush=True)
            predictor, training = fit(
                train, dev, model_seed, config, progress=lambda row: print(row, flush=True)
            )
            predictor.save(
                directory / "model.json",
                {"split_seed": split_seed, "model_seed": model_seed, "data_sha256": checksum},
            )
            # Held-out outcomes enter evaluation ONLY after fit has returned.
            report = {
                "training": training,
                "test_by_depth": evaluate(predictor, test),
                "baselines": baseline,
                "oracle_diagnostics": oracle_diagnostics(predictor, records),
            }
            save_json(directory / "report.json", report)
            summary.append(
                {
                    "run": name,
                    "best_epoch": training["best_epoch"],
                    "test_terminal": report["test_by_depth"]["6"],
                }
            )
            print(f"Completed {name}: terminal {report['test_by_depth']['6']}", flush=True)
    save_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
