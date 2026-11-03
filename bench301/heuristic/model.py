"""Compact prefix transformer with value, position, decision and cell embeddings."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bench301.representation import (
    ALLOWED_VALUES,
    CELL_TYPES,
    DECISION_TYPES,
    ENCODING,
    SEQUENCE_LENGTH,
    TOKEN_MAPPING,
    UNASSIGNED,
    VOCABULARY_SIZE,
)

MODEL_SPEC = {
    "kind": "darts-prefix-transformer-encoder",
    "sequence_length": SEQUENCE_LENGTH,
    "vocabulary_size": VOCABULARY_SIZE,
    "model_dimension": 32,
    "attention_heads": 4,
    "encoder_layers": 2,
    "feedforward_dimension": 64,
    "token_mapping": TOKEN_MAPPING,
}


class PrefixTransformer(nn.Module):
    def __init__(self, dropout=0.10):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout_rate = float(dropout)
        self.value_embedding = nn.Embedding(VOCABULARY_SIZE, 32)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH + 1, 32)
        self.decision_embedding = nn.Embedding(2, 32)
        self.cell_embedding = nn.Embedding(2, 32)
        self.summary_token = nn.Parameter(torch.empty(1, 1, 32))
        self.register_buffer("decision_types", torch.tensor(DECISION_TYPES, dtype=torch.long))
        self.register_buffer("cell_types", torch.tensor(CELL_TYPES, dtype=torch.long))
        legal = torch.zeros((SEQUENCE_LENGTH, VOCABULARY_SIZE), dtype=torch.bool)
        for position, values in enumerate(ALLOWED_VALUES):
            legal[position, list(values) + [UNASSIGNED]] = True
        self.register_buffer("legal_tokens", legal)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(32),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(32, 1)
        nn.init.normal_(self.summary_token, mean=0.0, std=0.02)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != SEQUENCE_LENGTH or tokens.dtype != torch.long:
            raise ValueError("Expected torch.long tokens shaped [batch, 22]")
        if torch.any((tokens < 0) | (tokens >= VOCABULARY_SIZE)):
            raise ValueError("Token outside NB301 vocabulary")
        positions = torch.arange(SEQUENCE_LENGTH, device=tokens.device)
        if not self.legal_tokens[positions[None, :], tokens].all():
            raise ValueError("Illegal token for DARTS decision position")
        padding = tokens == UNASSIGNED
        if torch.any(padding[:, :-1] & ~padding[:, 1:]):
            raise ValueError("UNASSIGNED tokens must form a suffix")
        values = (
            self.value_embedding(tokens)
            + self.position_embedding(positions + 1)
            + self.decision_embedding(self.decision_types)
            + self.cell_embedding(self.cell_types)
        )
        summary = self.summary_token.expand(tokens.shape[0], -1, -1)
        summary = summary + self.position_embedding(
            torch.zeros(1, dtype=torch.long, device=tokens.device)
        )
        sequence = torch.cat((summary, values), dim=1)
        mask = torch.cat(
            (torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device), padding),
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=mask)
        return self.output(encoded[:, 0]).squeeze(-1)


def _payload(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (payload.get("schema"), payload.get("benchmark"), payload.get("encoding")) != (
        1,
        "nasbench301",
        ENCODING,
    ):
        raise ValueError("Wrong benchmark/encoding for NB301 checkpoint")
    spec = payload.get("model", {})
    if any(spec.get(key) != value for key, value in MODEL_SPEC.items()):
        raise ValueError("NB301 transformer specification mismatch")
    metadata = payload.get("metadata", {})
    if not all(key in metadata for key in ("split_seed", "model_seed", "data_sha256")):
        raise ValueError("Missing checkpoint split/model/data provenance")
    return payload


def checkpoint_metadata(path):
    return _payload(path)["metadata"]


class FrozenPredictor:
    def __init__(self, model, mean, scale):
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid target normalization")
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite checkpoint weights")
        self.model = model.cpu().eval()
        self.model.requires_grad_(False)
        self.mean, self.scale = float(mean), float(scale)

    def predict_tokens(self, rows):
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != SEQUENCE_LENGTH or rows.dtype.kind not in {"i", "u"}:
            raise ValueError("Expected integer token matrix shaped [batch, 22]")
        with torch.inference_mode():
            result = (
                self.model(torch.from_numpy(rows.astype(np.int64, copy=False)))
                .numpy()
                .astype(np.float64)
            )
        result = result * self.scale + self.mean
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite predictor output")
        return result

    def predict(self, prefixes):
        from .data import tokenize

        return self.predict_tokens(
            np.asarray([tokenize(prefix) for prefix in prefixes], dtype=np.int64)
        )

    def __call__(self, prefix):
        return float(self.predict([prefix])[0])

    def save(self, path, metadata):
        payload = {
            "schema": 1,
            "benchmark": "nasbench301",
            "encoding": ENCODING,
            "model": {**MODEL_SPEC, "dropout": self.model.dropout_rate},
            "mean": self.mean,
            "scale": self.scale,
            "metadata": metadata,
            "weights": {key: value.tolist() for key, value in self.model.state_dict().items()},
        }
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False)

    @classmethod
    def load(cls, path):
        payload = _payload(path)
        model = PrefixTransformer(payload["model"]["dropout"])
        template = model.state_dict()
        weights = {
            key: torch.tensor(value, dtype=template[key].dtype)
            for key, value in payload["weights"].items()
        }
        for key in ("decision_types", "cell_types", "legal_tokens"):
            if not torch.equal(weights[key], template[key]):
                raise ValueError("Checkpoint changes fixed DARTS position semantics")
        model.load_state_dict(weights, strict=True)
        return cls(model, payload["mean"], payload["scale"])
