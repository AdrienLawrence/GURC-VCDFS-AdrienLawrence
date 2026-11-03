"""Parse official measured result JSON; never query a surrogate or use test accuracy."""

import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

from bench301.representation import BASE, ENCODING, config_to_state

SOURCE_VERSION = "NAS-Bench-301 measured dataset v1.0"
OFFICIAL_LOADER = (
    "https://github.com/automl/nasbench301/blob/master/nasbench301/surrogate_models/utils.py"
)


def percentage(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite percentage")
    if not 0 <= value <= 100:
        raise ValueError(f"{label} outside [0, 100]")
    return float(value)


def parse_record(raw, source_id):
    """Match ConfigLoader.__getitem__: summary val_accuracy, not final/test accuracy."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source_id}: result must be an object")
    try:
        config = raw["optimized_hyperparamater_config"]  # Official misspelling.
        state = config_to_state(config)
        if (
            raw["budget"],
            config["SimpleLearningrateSchedulerSelector:cosine_annealing:T_max"],
            config[BASE + "init_channels"],
            config[BASE + "layers"],
        ) != (100, 100, 32, 8):
            raise ValueError("Expected official 100-epoch/32-channel/8-layer fidelity")
        if not isinstance(raw["info"], list) or len(raw["info"]) != 1:
            raise ValueError("Expected one official result summary")
        info = raw["info"][0]
        if (info["train_datapoints"], info["val_datapoints"]) != (40000, 10000):
            raise ValueError("Unexpected CIFAR-10 train/validation data counts")
        if "cifar10" not in str(info["dataset_path"]).lower():
            raise ValueError("Unexpected objective dataset")
        epoch = info["epochs"]
        if type(epoch) is not int or not 0 <= epoch <= 100:
            raise ValueError("Invalid reported validation epoch")
        trial = {
            "source_id": source_id,
            "validation_accuracy": percentage(info["val_accuracy"], "val_accuracy"),
            "reported_epoch": epoch,
            "training_budget_epochs": 100,
            "validation_accuracy_final": percentage(
                info["val_accuracy_final"], "val_accuracy_final"
            ),
            "optimizer": raw.get("created_by_optimizer"),
            "seed": None,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source_id}: malformed official NB301 record: {error}") from error
    return state, trial


def iter_records(path):
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if Path(name).name.startswith("results_") and name.endswith(".json")
            )
            if not names:
                raise ValueError("Archive contains no official results_*.json records")
            for name in names:
                yield name, archive.read(name)
    elif path.is_dir():
        paths = sorted(path.rglob("results_*.json"))
        if not paths:
            raise ValueError("Directory contains no official results_*.json records")
        for file in paths:
            yield file.relative_to(path).as_posix(), file.read_bytes()
    else:
        raise FileNotFoundError(
            f"Missing official NB301 data archive/directory: {path}; run bench301.download"
        )


def extract(path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Measurements already exist: {output}")
    groups, sources = defaultdict(list), []
    for source_id, raw in iter_records(path):
        try:
            document = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{source_id}: invalid JSON") from error
        state, trial = parse_record(document, source_id)
        groups[state].append(trial)
        sources.append({"id": source_id, "sha256": hashlib.sha256(raw).hexdigest()})
    records = []
    for index, state in enumerate(sorted(groups)):
        trials = groups[state]
        records.append(
            {
                "index": index,
                "state": list(state),
                "trials": trials,
                "error": 100
                - math.fsum(trial["validation_accuracy"] for trial in trials) / len(trials),
            }
        )
    payload = {
        "schema": 1,
        "benchmark": "nasbench301",
        "source_version": SOURCE_VERSION,
        "encoding": ENCODING,
        "dataset": "cifar10",
        "metric": "info[0].val_accuracy",
        "units": "percentage_points",
        "aggregation": "available-observation-mean",
        "label_source": "real-measured-training-records",
        "sources": sources,
        "official_loader": OFFICIAL_LOADER,
        "epoch_convention": "100-epoch budget; official selected summary, not val_accuracy_final",
        "seed_convention": "Training seeds absent in release result records; source IDs retained",
        "record_count": len(records),
        "observation_count": len(sources),
        "duplicate_observations": len(sources) - len(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False)
    print(
        f"Extracted {len(sources)} real observations -> {len(records)} architectures -> {output}",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("bench301/artifacts/measurements.json"))
    args = parser.parse_args()
    extract(args.source, args.output)


if __name__ == "__main__":
    main()
