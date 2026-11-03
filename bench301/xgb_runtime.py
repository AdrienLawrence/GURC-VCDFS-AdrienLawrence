"""Memory-bounded v1.0 XGBoost inference using audited upstream encoding code.

Runs ONLY in the isolated Python 3.8 environment. It loads one member at a time.
ConfigLoader, fixed hyperparameters and genotype conversion come directly from
the pinned official source cached by setup_oracle. No surrogate is retrained.
"""

import ast
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import xgboost as xgb
from ConfigSpace.read_and_write import json as config_space_json_r_w

COMMIT = "61c670e60512abc14aa472da18bb8d5b985f95e0"


def upstream_code(upstream):
    upstream = Path(upstream)
    provenance = json.loads((upstream / "provenance.json").read_text())
    if provenance["commit"] != COMMIT:
        raise ValueError("Unexpected upstream inference revision")
    namespace = {
        "json": json,
        "os": os,
        "np": np,
        "CS": CS,
        "CSH": CSH,
        "config_space_json_r_w": config_space_json_r_w,
    }
    selected = (
        ("nasbench301/surrogate_models/utils.py", "ConfigLoader"),
        ("nasbench301/representations.py", "convert_genotype_to_config"),
        ("nasbench301/api.py", "fixed_hyperparameters"),
    )
    for relative, name in selected:
        path = upstream / relative
        text = path.read_text()
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["sources"][relative]:
            raise ValueError("Upstream encoding source checksum mismatch")
        tree = ast.parse(text, filename=str(path))
        nodes = [
            node
            for node in tree.body
            if getattr(node, "name", None) == name
            or isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        if len(nodes) != 1:
            raise ValueError("Required official encoding definition missing")
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
    return namespace


class _BoosterReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in {("xgboost.core", "Booster"), ("builtins", "bytearray")}:
            raise ValueError("Unexpected object in official Booster snapshot")
        return super().find_class(module, name)


class XGBEnsemble:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.upstream = self.directory.parents[1] / "upstream"
        definitions = upstream_code(self.upstream)
        self.convert = definitions["convert_genotype_to_config"]
        self.fixed = definitions["fixed_hyperparameters"]
        self.loader = definitions["ConfigLoader"](
            str(self.upstream / "nasbench301/configspace.json")
        )
        self.paths = sorted(self.directory.rglob("surrogate_model.model"))
        if len(self.paths) != 10:
            raise ValueError("Expected all ten official v1.0 XGBoost members")
        for path in self.paths:
            config = json.loads(path.with_name("model_config.json").read_text())
            if config["model"] != "xgb" or config["param:bananas_enc"]:
                raise ValueError("Expected ConfigSpace-encoded v1.0 XGBoost ensemble")

    def encode(self, genotype):
        config = {**self.fixed, **self.convert(genotype)}
        instance = self.loader.query_config_dict(config)
        values = instance.get_array().reshape(1, -1)
        values[np.isnan(values)] = -1
        return values

    def prepare_native(self, genotypes):
        """Convert training snapshots to inference files, proving prediction equality."""
        matrices = [xgb.DMatrix(self.encode(genotype), nthread=1) for genotype in genotypes]
        rows = []
        for path in self.paths:
            target = path.with_name("surrogate_inference.bin")
            if target.exists():
                rows.append({"member": str(path.relative_to(self.directory)), "existing": True})
                continue
            print("Preparing member " + str(path.relative_to(self.directory)), flush=True)
            with path.open("rb") as stream:
                booster = _BoosterReader(stream).load()
            booster.set_param({"nthread": 1})
            original = [float(booster.predict(matrix)[0]) for matrix in matrices]
            booster.save_model(str(target))
            del booster
            gc.collect()
            restored = xgb.Booster()
            restored.load_model(str(target))
            restored.set_param({"nthread": 1})
            converted = [float(restored.predict(matrix)[0]) for matrix in matrices]
            if original != converted:
                raise RuntimeError("Inference conversion changes official member predictions")
            rows.append(
                {
                    "member": str(path.relative_to(self.directory)),
                    "prediction_equality": True,
                    "predictions": original,
                    "inference_bytes": target.stat().st_size,
                }
            )
            del restored
            gc.collect()
        report = {
            "upstream_commit": COMMIT,
            "backend": "streamed-official-xgb",
            "xgboost": xgb.__version__,
            "ConfigSpace": CS.__version__,
            "member_count": len(self.paths),
            "noise": False,
            "members": rows,
        }
        report_path = self.directory / "inference_preparation.json"
        if not report_path.exists():
            report_path.write_text(json.dumps(report, indent=2))
        return report

    def predict(self, config, representation, with_noise):
        if representation != "genotype" or with_noise is not False:
            raise ValueError("Minimal XGBoost backend requires genotype and with_noise=False")
        matrix = xgb.DMatrix(self.encode(config), nthread=1)
        predictions = []
        for path in self.paths:
            target = path.with_name("surrogate_inference.bin")
            if not target.exists():
                raise FileNotFoundError("Run xgb_worker.py --prepare before search")
            booster = xgb.Booster()
            booster.load_model(str(target))
            booster.set_param({"nthread": 1})
            predictions.append(booster.predict(matrix))
            del booster
        return float(np.mean(predictions))
