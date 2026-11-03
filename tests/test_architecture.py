"""Architecture-boundary and neutral artifact-schema tests."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from schemas import (
    ArtifactEnvelope,
    SchemaValidationError,
    create_artifact,
    load_artifact,
    load_payload,
    write_artifact,
)

ROOT = Path(__file__).resolve().parent.parent
CORE_PATHS = (
    ROOT / "algs",
    ROOT / "bench201" / "heuristic",
    ROOT / "bench201" / "behavior",
    ROOT / "schemas",
    ROOT / "bench201" / "empirical.py",
    ROOT / "search_reporting.py",
    ROOT / "search_registry.py",
    ROOT / "bench201" / "specs.py",
    ROOT / "bench201" / "space.py",
    ROOT / "run.py",
    ROOT / "resource_measurement.py",
    ROOT / "bench301",
)


def python_files(path):
    return sorted(path.rglob("*.py")) if path.is_dir() else [path]


def imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.append(node.module.split(".", 1)[0])
    return roots


class DependencyBoundaryTests(unittest.TestCase):
    def test_algorithms_are_benchmark_neutral(self):
        forbidden = {"bench201", "bench301", "nas_space", "heuristic", "behavior"}
        for path in python_files(ROOT / "algs"):
            self.assertFalse(forbidden.intersection(imported_roots(path)), str(path))

    def test_experiment_code_never_imports_ui(self):
        violations = []
        for location in CORE_PATHS:
            for path in python_files(location):
                if "ui" in imported_roots(path):
                    violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [], f"core-to-UI imports are forbidden: {violations}")

    def test_neutral_schemas_do_not_import_project_packages(self):
        forbidden = {
            "algs",
            "heuristic",
            "ui",
            "empirical",
            "nas_space",
            "run",
            "bench201",
            "bench301",
        }
        violations = {}
        for path in python_files(ROOT / "schemas"):
            matches = sorted(forbidden.intersection(imported_roots(path)))
            if matches:
                violations[str(path.relative_to(ROOT))] = matches
        self.assertEqual(violations, {})


class ArtifactSchemaTests(unittest.TestCase):
    def result(self):
        return {
            "schema_version": 1,
            "artifact_kind": "run_result",
            "payload": {"run_id": "run-1", "algorithm": "VCDFS", "metrics": {"regret": 0.0}},
        }

    def trace(self):
        return {
            "schema_version": 1,
            "artifact_kind": "trace",
            "payload": {
                "run_id": "run-1",
                "events": [{"kind": "expand", "prefix": [0, 1]}],
                "trace_hash": "a" * 64,
            },
        }

    def test_valid_result_is_recursively_immutable(self):
        artifact = ArtifactEnvelope.from_dict(self.result())
        with self.assertRaises(TypeError):
            artifact.payload["algorithm"] = "GBFS"
        with self.assertRaises(TypeError):
            artifact.payload["metrics"]["regret"] = 1.0
        self.assertEqual(artifact.to_dict(), self.result())

    def test_loads_valid_trace_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(self.trace()), encoding="utf-8")
            artifact = load_artifact(path)
        self.assertEqual(artifact.artifact_kind, "trace")
        self.assertEqual(artifact.payload["events"][0]["kind"], "expand")

    def test_array_artifacts_round_trip(self):
        artifact = create_artifact("search_runs", [{"algorithm": "VCDFS"}])
        self.assertEqual(artifact.to_dict()["payload"], [{"algorithm": "VCDFS"}])

    def test_shared_writer_emits_versioned_envelopes(self):
        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "training.json"
            search_path = Path(directory) / "search.json"
            write_artifact(training_path, "heuristic_summary", [])
            write_artifact(search_path, "search_summary", [])
            training = load_artifact(training_path)
            search = load_artifact(search_path)
        self.assertEqual(
            (training.schema_version, training.artifact_kind), (1, "heuristic_summary")
        )
        self.assertEqual((search.schema_version, search.artifact_kind), (2, "search_summary"))

    def test_load_payload_checks_kind_and_explicit_legacy_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            legacy = {"train": [0], "dev": [1], "test": [2]}
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaises(SchemaValidationError):
                load_payload(path, "data_split")
            self.assertEqual(load_payload(path, "data_split", allow_legacy=True), legacy)

            path.write_text(
                json.dumps(create_artifact("data_split", legacy).to_dict()), encoding="utf-8"
            )
            self.assertEqual(load_payload(path, "data_split"), legacy)
            with self.assertRaises(SchemaValidationError):
                load_payload(path, "heuristic_report")

    def test_rejects_stale_unknown_and_malformed_artifacts(self):
        cases = []
        stale = self.result()
        stale["schema_version"] = 0
        cases.append(stale)
        unknown = self.result()
        unknown["artifact_kind"] = "mystery"
        cases.append(unknown)
        missing = self.result()
        del missing["payload"]["metrics"]
        cases.append(missing)
        extra = self.result()
        extra["created_by"] = "unversioned extension"
        cases.append(extra)
        nonfinite = self.result()
        nonfinite["payload"]["metrics"]["regret"] = float("nan")
        cases.append(nonfinite)
        for document in cases:
            with self.subTest(document=document), self.assertRaises(SchemaValidationError):
                ArtifactEnvelope.from_dict(document)

    def test_rejects_malformed_trace(self):
        bad_event = self.trace()
        bad_event["payload"]["events"] = [{"prefix": []}]
        bad_hash = self.trace()
        bad_hash["payload"]["trace_hash"] = "not-a-hash"
        for document in (bad_event, bad_hash):
            with self.subTest(document=document), self.assertRaises(SchemaValidationError):
                ArtifactEnvelope.from_dict(document)


if __name__ == "__main__":
    unittest.main()
