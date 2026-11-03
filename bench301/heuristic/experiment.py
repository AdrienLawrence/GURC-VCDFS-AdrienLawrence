"""Train NB301 frozen prefix models on real records; no online searches."""

if __name__ == "__main__" and not __package__:
    raise SystemExit(
        "Open bench301/runs/train_bench301.py and press Run. "
        "This implementation module needs package imports and study settings."
    )

import argparse
import hashlib
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from bench301.representation import ENCODING, TOKEN_MAPPING
from schemas import load_payload, write_artifact

from .data import load_measurements, select_records, split_indices
from .evaluate import evaluate
from .model import checkpoint_metadata
from .train import TrainingConfig, fit


def source_digest():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = [
        *sorted((root / "bench301").glob("*.py")),
        *sorted((root / "bench301" / "heuristic").glob("*.py")),
        *sorted((root / "bench301" / "behavior").glob("*.py")),
        *sorted((root / "algs").glob("*.py")),
        root / "search_registry.py",
    ]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("bench301/artifacts/measurements.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--dev-size", type=int)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if any(len(values) != len(set(values)) for values in (args.split_seeds, args.model_seeds)):
        parser.error("Seeds must be unique")
    config = TrainingConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
    records, checksum = load_measurements(args.data)
    manifest = {
        "benchmark": "nasbench301",
        "encoding": ENCODING,
        "data_sha256": checksum,
        "source_sha256": source_digest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "split_seeds": args.split_seeds,
        "model_seeds": args.model_seeds,
        "train_size": args.train_size,
        "dev_size": args.dev_size,
        "default_split": "80/10/10 complete canonical architectures",
        "training_config": asdict(config),
        "measured_architectures": len(records),
        "labels": "real measured validation only; never surrogate/test labels",
        "target_scope": "empirical mean under measured-data sampling distribution, not uniform full-space expectation",
    }
    if args.output.exists():
        if not args.output.is_dir() or not (args.output / "manifest.json").is_file():
            raise ValueError("Existing training output has no resumable manifest")
        saved = load_payload(args.output / "manifest.json", "nb301_heuristic_manifest")
        if saved != manifest:
            raise ValueError("Cannot resume NB301 training with a changed data/code/config plan")
    else:
        args.output.mkdir(parents=True)
        write_artifact(args.output / "manifest.json", "nb301_heuristic_manifest", manifest)
    summary = []
    for split_seed in args.split_seeds:
        split = split_indices(len(records), split_seed, args.train_size, args.dev_size)
        train, dev, test = [select_records(records, split[key]) for key in ("train", "dev", "test")]
        for model_seed in args.model_seeds:
            name = f"split-{split_seed}-model-{model_seed}"
            directory = args.output / name
            if directory.exists():
                if not directory.is_dir() or not (directory / "split.json").is_file():
                    raise ValueError(f"Incomplete training split metadata: {directory}")
                if load_payload(directory / "split.json", "data_split") != split:
                    raise ValueError(f"Training split changed for {name}")
            else:
                directory.mkdir()
                write_artifact(directory / "split.json", "data_split", split)
            model_path, report_path = directory / "model.json", directory / "report.json"
            if model_path.exists() != report_path.exists():
                raise ValueError(
                    f"Interrupted checkpoint write for {name}; inspect its model/report files"
                )
            if model_path.exists():
                metadata = checkpoint_metadata(model_path)
                if (
                    metadata["split_seed"] != split_seed
                    or metadata["model_seed"] != model_seed
                    or metadata["data_sha256"] != checksum
                ):
                    raise ValueError(f"Checkpoint identity changed for {name}")
                report = load_payload(report_path, "nb301_heuristic_report")
                row = {
                    "run": name,
                    "best_epoch": report["training"]["best_epoch"],
                    "terminal": report["heldout"]["by_depth"]["22"],
                }
                summary.append(row)
                print(f"RESUME | completed {name}", flush=True)
                continue
            print(
                f"Training NB301 {name}: train={len(train)} dev={len(dev)} test={len(test)}",
                flush=True,
            )
            predictor, training = fit(
                train, dev, model_seed, config, progress=lambda row: print(row, flush=True)
            )
            metadata = {
                "benchmark": "nasbench301",
                "encoding": ENCODING,
                "token_mapping": TOKEN_MAPPING,
                "split_seed": split_seed,
                "model_seed": model_seed,
                "data_sha256": checksum,
                "train_architectures": len(train),
                "offline_prefix_examples": len(train) * 22,
            }
            predictor.save(model_path, metadata)
            # The held-out partition is first presented after fitting/restoration.
            report = {"training": training, "heldout": evaluate(predictor, test)}
            write_artifact(report_path, "nb301_heuristic_report", report)
            row = {
                "run": name,
                "best_epoch": training["best_epoch"],
                "terminal": report["heldout"]["by_depth"]["22"],
            }
            summary.append(row)
            print(f"Completed {name}: {row['terminal']}", flush=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        if load_payload(summary_path, "nb301_heuristic_summary") != summary:
            raise ValueError("Existing NB301 training summary disagrees with completed checkpoints")
    else:
        write_artifact(summary_path, "nb301_heuristic_summary", summary)


if __name__ == "__main__":
    main()
