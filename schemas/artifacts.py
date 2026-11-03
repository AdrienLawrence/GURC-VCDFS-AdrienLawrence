"""Immutable JSON artifact envelopes shared by producers and read-only consumers.

This module deliberately imports only the Python standard library. Experiment
code may produce these envelopes and downstream interfaces may read them, but
neither side needs to import the other.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

RESULT_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1

_SUPPORTED_VERSIONS = {
    "data_split": 1,
    "heuristic_manifest": 1,
    "heuristic_report": 1,
    "heuristic_summary": 1,
    "run_result": RESULT_SCHEMA_VERSION,
    "search_manifest": 2,
    "search_runs": 2,
    "search_split_summary": 2,
    "search_summary": 2,
    "behavior_manifest": 1,
    "behavior_runs": 1,
    "behavior_split_summary": 1,
    "behavior_summary": 1,
    "trace": TRACE_SCHEMA_VERSION,
    "nb301_heuristic_manifest": 1,
    "nb301_heuristic_report": 1,
    "nb301_heuristic_summary": 1,
    "nb301_search_manifest": 1,
    "nb301_search_runs": 1,
    "nb301_search_summary": 1,
    "nb301_reference": 1,
    "nb301_behavior_manifest": 1,
    "nb301_behavior_runs": 1,
    "nb301_behavior_split_summary": 1,
    "nb301_behavior_summary": 1,
}


class SchemaValidationError(ValueError):
    """Raised when an artifact is malformed, unknown, or stale."""


def _freeze_json(value, path="payload"):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise SchemaValidationError(f"{path} contains a non-string object key")
        return MappingProxyType(
            {key: _freeze_json(item, f"{path}.{key}") for key, item in value.items()}
        )
    raise SchemaValidationError(f"{path} contains unsupported type {type(value).__name__}")


def _thaw_json(value):
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_string(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"payload.{key} must be a nonempty string")


def _validate_result(payload):
    _require_string(payload, "run_id")
    _require_string(payload, "algorithm")
    if not isinstance(payload.get("metrics"), Mapping):
        raise SchemaValidationError("payload.metrics must be an object")


def _validate_trace(payload):
    _require_string(payload, "run_id")
    events = payload.get("events")
    if not isinstance(events, tuple):
        raise SchemaValidationError("payload.events must be an array")
    if not all(
        isinstance(event, Mapping) and isinstance(event.get("kind"), str) for event in events
    ):
        raise SchemaValidationError("every trace event must be an object with a string kind")
    trace_hash = payload.get("trace_hash")
    if (
        not isinstance(trace_hash, str)
        or len(trace_hash) != 64
        or any(character not in "0123456789abcdef" for character in trace_hash)
    ):
        raise SchemaValidationError("payload.trace_hash must be a lowercase SHA-256 hex digest")


def _validate_object(payload):
    if not isinstance(payload, Mapping):
        raise SchemaValidationError("payload must be an object")


def _validate_array(payload):
    if not isinstance(payload, tuple):
        raise SchemaValidationError("payload must be an array")


def _validate_split(payload):
    _validate_object(payload)
    if set(payload) != {"train", "dev", "test"}:
        raise SchemaValidationError("data_split payload must contain train, dev, and test")
    if not all(isinstance(payload[key], tuple) for key in ("train", "dev", "test")):
        raise SchemaValidationError("data_split partitions must be arrays")


def _validate_nb301_behavior_runs(payload):
    _validate_array(payload)
    required = {
        "checkpoint",
        "algorithm",
        "query_budget",
        "stop",
        "found",
        "queries",
        "expanded",
        "scored",
        "best_error",
        "best_so_far_curve",
    }
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or not required <= row.keys():
            raise SchemaValidationError(f"NB301 behavior row {index} lacks required metrics")
        if not all(
            isinstance(row[key], str) and row[key] for key in ("checkpoint", "algorithm", "stop")
        ):
            raise SchemaValidationError(f"NB301 behavior row {index} has invalid identity")
        if type(row["query_budget"]) is not int or row["query_budget"] == 0:
            raise SchemaValidationError(f"NB301 behavior row {index} has invalid budget")
        if type(row["found"]) is not bool:
            raise SchemaValidationError(f"NB301 behavior row {index} has invalid found flag")
        if any(
            type(row[key]) is not int or row[key] < 0 for key in ("queries", "expanded", "scored")
        ):
            raise SchemaValidationError(f"NB301 behavior row {index} has invalid work counts")
        if not isinstance(row["best_so_far_curve"], tuple):
            raise SchemaValidationError(f"NB301 behavior row {index} has invalid improvement curve")
        value = row["best_error"]
        if row["found"] != (value is not None and isinstance(value, (int, float))):
            raise SchemaValidationError(f"NB301 behavior row {index} has inconsistent quality")
        if row["found"] and (row["queries"] < 1 or not row["best_so_far_curve"]):
            raise SchemaValidationError(f"NB301 behavior row {index} lacks a terminal event")
        if not row["found"] and (row["queries"] or row["best_so_far_curve"]):
            raise SchemaValidationError(f"NB301 behavior row {index} has a missing best result")


def _validate_nb301_behavior_summary(payload):
    _validate_array(payload)
    for index, row in enumerate(payload):
        if (
            not isinstance(row, Mapping)
            or not {
                "algorithm",
                "query_budget",
                "model_runs",
                "valid_runs",
            }
            <= row.keys()
        ):
            raise SchemaValidationError(f"NB301 behavior summary row {index} is incomplete")
        if type(row["valid_runs"]) is not int or type(row["model_runs"]) is not int:
            raise SchemaValidationError(f"NB301 behavior summary row {index} has invalid counts")
        if not 0 <= row["valid_runs"] <= row["model_runs"]:
            raise SchemaValidationError(f"NB301 behavior summary row {index} has impossible counts")


_VALIDATORS = {
    "data_split": _validate_split,
    "heuristic_manifest": _validate_object,
    "heuristic_report": _validate_object,
    "heuristic_summary": _validate_array,
    "run_result": _validate_result,
    "search_manifest": _validate_object,
    "search_runs": _validate_array,
    "search_split_summary": _validate_array,
    "search_summary": _validate_array,
    "behavior_manifest": _validate_object,
    "behavior_runs": _validate_array,
    "behavior_split_summary": _validate_array,
    "behavior_summary": _validate_array,
    "trace": _validate_trace,
    "nb301_heuristic_manifest": _validate_object,
    "nb301_heuristic_report": _validate_object,
    "nb301_heuristic_summary": _validate_array,
    "nb301_search_manifest": _validate_object,
    "nb301_search_runs": _validate_array,
    "nb301_search_summary": _validate_array,
    "nb301_reference": _validate_object,
    "nb301_behavior_manifest": _validate_object,
    "nb301_behavior_runs": _validate_nb301_behavior_runs,
    "nb301_behavior_split_summary": _validate_nb301_behavior_summary,
    "nb301_behavior_summary": _validate_nb301_behavior_summary,
}


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    """Validated immutable view of one saved artifact."""

    schema_version: int
    artifact_kind: str
    payload: Any

    @classmethod
    def from_dict(cls, document):
        if not isinstance(document, dict):
            raise SchemaValidationError("artifact must be a JSON object")
        required = {"schema_version", "artifact_kind", "payload"}
        if set(document) != required:
            raise SchemaValidationError(f"artifact fields must be exactly {sorted(required)}")
        kind = document["artifact_kind"]
        if kind not in _SUPPORTED_VERSIONS:
            raise SchemaValidationError(f"unsupported artifact kind: {kind!r}")
        version = document["schema_version"]
        expected = _SUPPORTED_VERSIONS[kind]
        if type(version) is not int or version != expected:
            raise SchemaValidationError(
                f"unsupported {kind} schema version {version!r}; expected {expected}"
            )
        payload = _freeze_json(document["payload"])
        _VALIDATORS[kind](payload)
        return cls(version, kind, payload)

    def to_dict(self):
        """Return a detached mutable copy suitable for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "payload": _thaw_json(self.payload),
        }


def load_artifact(path):
    """Read and validate an artifact without granting mutation of its contents."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaValidationError(f"cannot read artifact: {error}") from error
    return ArtifactEnvelope.from_dict(document)


def create_artifact(artifact_kind, payload):
    """Validate and freeze a new artifact using the current kind-specific version."""
    if artifact_kind not in _SUPPORTED_VERSIONS:
        raise SchemaValidationError(f"unsupported artifact kind: {artifact_kind!r}")
    return ArtifactEnvelope.from_dict(
        {
            "schema_version": _SUPPORTED_VERSIONS[artifact_kind],
            "artifact_kind": artifact_kind,
            "payload": payload,
        }
    )


def write_artifact(path, artifact_kind, payload, *, replace=False):
    """Write one validated artifact, optionally replacing it atomically."""
    document = create_artifact(artifact_kind, payload).to_dict()
    path = Path(path)
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def load_payload(path, expected_kind, *, allow_legacy=False):
    """Load a detached payload, optionally accepting a pre-envelope artifact."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaValidationError(f"cannot read artifact: {error}") from error
    if isinstance(document, dict) and {
        "schema_version",
        "artifact_kind",
        "payload",
    }.issubset(document):
        artifact = ArtifactEnvelope.from_dict(document)
        if artifact.artifact_kind != expected_kind:
            raise SchemaValidationError(
                f"expected {expected_kind!r}, found {artifact.artifact_kind!r}"
            )
        return artifact.to_dict()["payload"]
    if allow_legacy:
        return document
    raise SchemaValidationError("artifact has no versioned envelope")
