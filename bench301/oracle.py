"""Deterministic complete-architecture oracle with lazy optional legacy loading."""

import hashlib
import json
import math
import os
import queue
import subprocess
import threading
from pathlib import Path

from .representation import state_to_genotype


def model_digest(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing NB301 v1.0 performance ensemble: {directory}; download models explicitly"
        )
    files = sorted(directory.rglob("*"))
    if not any(path.name.endswith("surrogate_model.model") for path in files):
        raise FileNotFoundError(f"No surrogate_model.model ensemble members in {directory}")
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


class _Bridge:
    """One persistent JSONL worker, using an independently installed runtime."""

    def __init__(self, python, directory, worker="oracle_bridge.py"):
        self.process = subprocess.Popen(
            [
                str(python),
                "-u",
                str(Path(__file__).with_name(worker)),
                "--model",
                str(Path(directory).resolve()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        line = self._receive(timeout=120, stage="startup")
        if not line:
            self.close()
            raise RuntimeError(
                "NB301 worker failed to initialize; see stderr and oracle runtime setup"
            )
        response = json.loads(line)
        if response.get("ready") is not True:
            self.close()
            raise RuntimeError(f"NB301 worker initialization failed: {response}")

    def _read_output(self):
        try:
            for line in self.process.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _receive(self, timeout, stage):
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty as error:
            self.process.terminate()
            self.process.wait(timeout=10)
            raise TimeoutError(f"NB301 oracle worker exceeded {timeout}s during {stage}") from error

    def predict(self, config, representation, with_noise):
        request = {"genotype": config._asdict(), "with_noise": with_noise}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self._receive(timeout=90, stage="prediction")
        if not line:
            raise RuntimeError("NB301 oracle worker exited during prediction")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"NB301 worker error: {response['error']}")
        return response["accuracy"]

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        if self.process.stdout:
            self.process.stdout.close()


class TerminalOracle:
    """Official v1.0 predicted validation accuracy -> percentage-point error."""

    def __init__(self, surrogate, provenance=None):
        self.surrogate = surrogate
        self.provenance = provenance or {
            "benchmark": "nasbench301",
            "scope": "injected test oracle",
        }
        self.cache = {}
        self.surrogate_calls = 0

    def __call__(self, state):
        genotype = state_to_genotype(state)  # Also rejects incomplete/illegal states.
        if state not in self.cache:
            value = self.surrogate.predict(
                config=genotype, representation="genotype", with_noise=False
            )
            accuracy = float(value.item() if hasattr(value, "item") else value)
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
                raise ValueError("NB301 surrogate produced invalid validation accuracy percentage")
            self.cache[state] = 100.0 - accuracy
            self.surrogate_calls += 1
        return self.cache[state]

    def clear(self):
        self.cache.clear()
        self.surrogate_calls = 0

    def close(self):
        if isinstance(self.surrogate, _Bridge):
            self.surrogate.close()

    @classmethod
    def load(cls, directory, python=None, *, backend="official"):
        if backend not in {"official", "minimal-xgb"}:
            raise ValueError("Unknown NB301 oracle backend")
        directory = Path(directory)
        checksum = model_digest(directory)
        # v1.0 XGBoost performance ensemble, never a runtime model.
        members = sorted(directory.rglob("model_config.json"))
        if not members or any(json.loads(path.read_text())["model"] != "xgb" for path in members):
            raise ValueError("Expected official v1.0 XGBoost performance ensemble")
        if directory.name != "xgb_v1.0":
            raise ValueError("Use the official xgb_v1.0 directory to identify the verified release")
        python = python or os.environ.get("NB301_PYTHON")
        if backend == "minimal-xgb" and not python:
            raise ValueError("minimal-xgb requires the isolated interpreter from setup_oracle")
        if python:
            worker = "xgb_worker.py" if backend == "minimal-xgb" else "oracle_bridge.py"
            surrogate = _Bridge(python, directory, worker)
            runtime = (
                "streamed-official-xgb" if backend == "minimal-xgb" else "isolated-jsonl-worker"
            )
        else:
            try:
                import nasbench301
            except ImportError as error:
                raise RuntimeError(
                    "Official NB301 inference dependencies are unavailable. Set NB301_PYTHON "
                    "to an independently prepared legacy runtime; keep the current environment unchanged."
                ) from error
            surrogate = nasbench301.load_ensemble(str(directory))
            runtime = "direct-optional-runtime"
        return cls(
            surrogate,
            {
                "benchmark": "nasbench301",
                "release": "v1.0",
                "model": "xgb",
                "model_sha256": checksum,
                "runtime": runtime,
                "backend": backend,
                "with_noise": False,
                "target": "validation accuracy percentage (official ConfigLoader val_accuracy)",
                "objective": "100 - deterministic predicted validation accuracy",
            },
        )
