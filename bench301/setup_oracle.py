"""Explicit local setup for the verified v1.0 XGBoost oracle; primary venv untouched."""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM_COMMIT = "61c670e60512abc14aa472da18bb8d5b985f95e0"
SOURCES = (
    "nasbench301/configspace.json",
    "nasbench301/api.py",
    "nasbench301/representations.py",
    "nasbench301/surrogate_models/utils.py",
    "nasbench301/surrogate_models/ensemble.py",
    "nasbench301/surrogate_models/gradient_boosting/xgboost.py",
)
PACKAGES = (
    "numpy==1.19.5",
    "scipy==1.6.3",
    "ConfigSpace==0.4.21",
    "xgboost==1.2.1",
    "pyparsing==2.4.7",
    "Cython==0.29.37",
)
PYTHON_URL = "https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-amd64.zip"


def fetch(url, target):
    target = Path(target)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, target.open("xb") as stream:
        shutil.copyfileobj(response, stream)
    return target


def prepare(archive, cache):
    archive, cache = Path(archive), Path(cache)
    if not archive.is_file():
        raise FileNotFoundError(f"Missing supplied NB301 model archive: {archive}")
    digest = hashlib.md5()
    with archive.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != "2cf9845823c67e323a5905a0321bfec8":
        raise ValueError("Model archive does not match the published v1.0 MD5")
    model_root = cache / "nb_models_1.0" / "xgb_v1.0"
    model_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        members = [
            name
            for name in zip_file.namelist()
            if "/xgb_v1.0/" in name
            and Path(name).name
            in {"surrogate_model.model", "model_config.json", "data_config.json"}
        ]
        if sum(name.endswith("surrogate_model.model") for name in members) != 10:
            raise ValueError("Expected ten official XGBoost members")
        for name in members:
            relative = Path(name.split("/xgb_v1.0/", 1)[1])
            target = (model_root / relative).resolve()
            if not target.is_relative_to(model_root.resolve()):
                raise ValueError("Unsafe model archive path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != zip_file.getinfo(name).file_size:
                    raise ValueError(f"Incomplete existing extracted file: {target}")
                continue
            print(f"Extracting {relative}", flush=True)
            with zip_file.open(name) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    upstream = cache / "upstream"
    provenance = {}
    for relative in SOURCES:
        url = f"https://raw.githubusercontent.com/automl/nasbench301/{UPSTREAM_COMMIT}/{relative}"
        path = fetch(url, upstream / relative)
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance_path = upstream / "provenance.json"
    if not provenance_path.exists():
        provenance_path.write_text(
            json.dumps({"commit": UPSTREAM_COMMIT, "sources": provenance}, indent=2)
        )
    runtime = cache / "oracle-runtime"
    python = runtime / "python.exe"
    if not python.exists():
        bundle = fetch(PYTHON_URL, cache / "python-3.8.10-embed-amd64.zip")
        runtime.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle) as zip_file:
            for member in zip_file.infolist():
                target = (runtime / member.filename).resolve()
                if not target.is_relative_to(runtime.resolve()):
                    raise ValueError("Unsafe Python runtime archive path")
            zip_file.extractall(runtime)
        (runtime / "python38._pth").write_text(
            "python38.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8"
        )
    bootstrap = fetch("https://bootstrap.pypa.io/pip/3.8/get-pip.py", cache / "get-pip-3.8.py")
    subprocess.run([str(python), str(bootstrap)], check=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--only-binary=:all:", *PACKAGES],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-u",
            str(Path(__file__).with_name("xgb_worker.py")),
            "--model",
            str(model_root.resolve()),
            "--prepare",
        ],
        check=True,
    )
    print(f"NB301 interpreter: {python.resolve()}", flush=True)
    print(f"NB301 model: {model_root.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("bench301/cache"))
    args = parser.parse_args()
    prepare(args.archive, args.cache)


if __name__ == "__main__":
    main()
