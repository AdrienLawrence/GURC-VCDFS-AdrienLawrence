"""Deterministic, batched training; fit receives only train and development records."""

import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PrefixDataset, validate_records
from .model import FrozenPredictor, PrefixTransformer


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    patience: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.epochs, self.patience, self.batch_size)
        ):
            raise ValueError("Training counts must be positive integers")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def _mse(model, loader):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for tokens, targets in loader:
            total += float(((model(tokens) - targets) ** 2).sum())
            count += len(targets)
    return total / count


def fit(train_records, dev_records, seed=0, config=TrainingConfig(), progress=None):
    validate_records(train_records)
    validate_records(dev_records)
    if {tuple(row["state"]) for row in train_records} & {
        tuple(row["state"]) for row in dev_records
    }:
        raise ValueError("Training and development architectures overlap")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    errors = np.asarray([row["error"] for row in train_records], dtype=np.float64)
    mean, scale = float(errors.mean()), float(errors.std())
    if scale < 1e-12:
        scale = 1.0
    train = PrefixDataset(train_records, mean, scale)
    dev = PrefixDataset(dev_records, mean, scale)
    generator = torch.Generator().manual_seed(seed)
    batches = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator)
    train_eval = DataLoader(train, batch_size=config.batch_size)
    dev_eval = DataLoader(dev, batch_size=config.batch_size)
    model = PrefixTransformer()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, best_state = [], None
    best_loss, best_epoch, stale = float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        for tokens, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tokens) - targets) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        train_loss, dev_loss = _mse(model, train_eval), _mse(model, dev_eval)
        if not np.isfinite(train_loss) or not np.isfinite(dev_loss):
            raise ValueError("Nonfinite evaluation loss")
        row = {
            "epoch": epoch,
            "train_mse_pp2": train_loss * scale**2,
            "dev_mse_pp2": dev_loss * scale**2,
        }
        history.append(row)
        if dev_loss < best_loss:
            best_loss, best_epoch, best_state, stale = (
                dev_loss,
                epoch,
                deepcopy(model.state_dict()),
                0,
            )
        else:
            stale += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            progress(row)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return FrozenPredictor(model, mean, scale), {
        "config": asdict(config),
        "initialization_seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "normalization": {"mean": mean, "scale": scale},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_architectures": len(train_records),
        "dev_architectures": len(dev_records),
        "prefix_examples": len(train),
        "train_seconds": time.perf_counter() - started,
    }
