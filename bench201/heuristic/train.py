"""Fit the frozen predictor. This function receives NO held-out records."""

import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import examples
from .model import FrozenPredictor, PrefixTransformer


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    patience: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self):
        for value in (self.epochs, self.patience, self.batch_size):
            if type(value) is not int or value < 1:
                raise ValueError("Epoch, patience and batch counts must be positive integers")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def fit(train_records, dev_records, seed=0, config=TrainingConfig(), progress=None):
    train_states = {tuple(r["state"]) for r in train_records}
    if train_states & {tuple(r["state"]) for r in dev_records}:
        raise ValueError("Training and development architectures overlap")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    x, y, _ = examples(train_records)
    dx, dy, _ = examples(dev_records)
    mean, scale = float(y.mean()), float(y.std())
    if scale < 1e-12:
        scale = 1.0
    tx, ty = torch.from_numpy(x), torch.tensor((y - mean) / scale, dtype=torch.float32)
    vx, vy = torch.from_numpy(dx), torch.tensor((dy - mean) / scale, dtype=torch.float32)
    model = PrefixTransformer()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    history, best_state = [], None
    best_loss, best_epoch, stale = float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(tx), generator=generator)
        for batch in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx[batch]) - ty[batch]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_loss = float(((model(tx) - ty) ** 2).mean())
            dev_loss = float(((model(vx) - vy) ** 2).mean())
        if not np.isfinite(train_loss) or not np.isfinite(dev_loss):
            raise ValueError("Nonfinite epoch metrics")
        history.append(
            {
                "epoch": epoch,
                "train_mse_pp2": train_loss * scale**2,
                "dev_mse_pp2": dev_loss * scale**2,
            }
        )
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            progress(history[-1])
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    report = {
        "config": asdict(config),
        "initialization_seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train_seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "normalization": {"mean": mean, "scale": scale},
        "history": history,
    }
    return FrozenPredictor(model, mean, scale), report
