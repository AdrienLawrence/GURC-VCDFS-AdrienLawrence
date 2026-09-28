"""Compact encoder-only transformer for six-edge NAS prefixes."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEQUENCE_LENGTH = 6
VOCABULARY_SIZE = 6
MODEL_DIMENSION = 32


class PrefixTransformer(nn.Module):
    """Operation tokens -> contextual edge representations -> scalar error."""

    def __init__(self, dropout=0.10):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.dropout_rate = float(dropout)
        self.operation_embedding = nn.Embedding(VOCABULARY_SIZE, MODEL_DIMENSION)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH + 1, MODEL_DIMENSION)
        self.summary_token = nn.Parameter(torch.empty(1, 1, MODEL_DIMENSION))
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIMENSION,
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
            norm=nn.LayerNorm(MODEL_DIMENSION),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(MODEL_DIMENSION, 1)
        nn.init.normal_(self.summary_token, mean=0.0, std=0.02)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != SEQUENCE_LENGTH:
            raise ValueError("Expected token matrix shaped [batch, 6]")
        if tokens.dtype != torch.long:
            raise ValueError("Architecture tokens must use torch.long")
        if torch.any((tokens < 0) | (tokens >= VOCABULARY_SIZE)):
            raise ValueError("Architecture token outside [0, 5]")
        batch = tokens.shape[0]
        positions = torch.arange(1, SEQUENCE_LENGTH + 1, device=tokens.device)
        edges = self.operation_embedding(tokens) + self.position_embedding(positions)
        summary = self.summary_token.expand(batch, -1, -1) + self.position_embedding(
            torch.zeros(1, dtype=torch.long, device=tokens.device)
        )
        sequence = torch.cat((summary, edges), dim=1)
        padding = torch.cat(
            (torch.zeros((batch, 1), dtype=torch.bool, device=tokens.device), tokens == 5),
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.output(encoded[:, 0]).squeeze(-1)


class FrozenPredictor:
    """Read-only transformer plus TRAIN-only target normalization."""

    def __init__(self, model, mean, scale):
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid target normalization")
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("Nonfinite checkpoint weights")
        self.model = model.cpu().eval()
        self.model.requires_grad_(False)
        self.mean, self.scale = float(mean), float(scale)

    def predict(self, prefixes):
        from .data import tokenize

        rows = np.asarray([tokenize(prefix) for prefix in prefixes], dtype=np.int64)
        return self.predict_tokens(rows)

    def predict_tokens(self, rows):
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != SEQUENCE_LENGTH:
            raise ValueError("Expected token matrix shaped [batch, 6]")
        if rows.dtype.kind not in {"i", "u"} or np.any((rows < 0) | (rows >= VOCABULARY_SIZE)):
            raise ValueError("Token matrix must contain integer IDs in [0, 5]")
        rows = rows.astype(np.int64, copy=False)
        with torch.inference_mode():
            result = self.model(torch.from_numpy(rows)).numpy().astype(np.float64)
        result = result * self.scale + self.mean
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite model prediction")
        return result

    def __call__(self, prefix):
        return float(self.predict([prefix])[0])

    def save(self, path, metadata):
        payload = {
            "schema": 2,
            "model": {
                "kind": "prefix-transformer-encoder",
                "sequence_length": 6,
                "vocabulary_size": 6,
                "model_dimension": 32,
                "attention_heads": 4,
                "encoder_layers": 2,
                "feedforward_dimension": 64,
                "dropout": self.model.dropout_rate,
            },
            "mean": self.mean,
            "scale": self.scale,
            "metadata": metadata,
            "weights": {key: value.tolist() for key, value in self.model.state_dict().items()},
        }
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        specification = payload.get("model", {})
        if payload.get("schema") != 2 or specification.get("kind") != "prefix-transformer-encoder":
            raise ValueError("Unsupported checkpoint")
        expected = {
            "sequence_length": 6,
            "vocabulary_size": 6,
            "model_dimension": 32,
            "attention_heads": 4,
            "encoder_layers": 2,
            "feedforward_dimension": 64,
        }
        if any(specification.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint transformer specification does not match")
        dropout = specification.get("dropout")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("Invalid checkpoint dropout")
        model = PrefixTransformer(dropout=dropout)
        model.load_state_dict(
            {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in payload["weights"].items()
            },
            strict=True,
        )
        return cls(model, payload["mean"], payload["scale"])
