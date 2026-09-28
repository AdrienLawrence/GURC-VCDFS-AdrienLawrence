"""Post-fit diagnostics and transparent non-neural prediction baselines."""

from collections import defaultdict

import numpy as np

from .data import examples


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def metrics(predicted, actual):
    p, y = np.asarray(predicted), np.asarray(actual)
    if len(p) == 0 or p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Invalid evaluation arrays")
    rp, ry = ranks(p), ranks(y)
    correlation = None if rp.std() == 0 or ry.std() == 0 else float(np.corrcoef(rp, ry)[0, 1])
    return {
        "n": len(y),
        "mae_pp": float(np.abs(p - y).mean()),
        "rmse_pp": float(np.sqrt(((p - y) ** 2).mean())),
        "spearman": correlation,
    }


def evaluate(predictor, records):
    x, y, depths = examples(records)
    p = predictor.predict_tokens(x)
    report = {str(d): metrics(p[depths == d], y[depths == d]) for d in range(1, 7)}
    terminal = depths == 6
    cutoff = float(np.quantile(y[terminal], 0.1))
    tail = terminal & (y <= cutoff)
    report["best_decile_terminal"] = metrics(p[tail], y[tail])
    report["best_decile_cutoff_pp"] = cutoff
    report["out_of_range_predictions"] = int(((p < 0) | (p > 100)).sum())
    return report


class Baselines:
    """All fitted only on training outcomes. Ridge strength is fixed at 1."""

    def __init__(self, records):
        x, y, _ = examples(records)
        self.mean = float(y.mean())
        sums, counts = defaultdict(float), defaultdict(int)
        for row in records:
            state = tuple(row["state"])
            for depth in range(7):
                sums[state[:depth]] += row["error"]
                counts[state[:depth]] += 1
        self.lookup = {p: sums[p] / count for p, count in counts.items()}
        one_hot = np.eye(6, dtype=np.float64)[x].reshape(len(x), 36)
        design = np.column_stack([np.ones(len(x)), one_hot])
        penalty = np.eye(37)
        penalty[0, 0] = 0
        self.linear = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    def predict(self, x, kind):
        x = np.asarray(x, dtype=np.int64)
        if kind == "constant":
            return np.full(len(x), self.mean)
        if kind == "linear":
            one_hot = np.eye(6, dtype=np.float64)[x].reshape(len(x), 36)
            return np.column_stack([np.ones(len(x)), one_hot]) @ self.linear
        if kind != "prefix":
            raise ValueError("Unknown baseline")
        predictions = []
        for row in x:
            values = row.tolist()
            prefix = tuple(values[: values.index(5)] if 5 in values else values)
            while prefix not in self.lookup:
                prefix = prefix[:-1]
            predictions.append(self.lookup[prefix])
        return np.asarray(predictions)


def baseline_evaluation(baselines, records):
    x, y, depths = examples(records)
    report = {}
    for kind in ("constant", "prefix", "linear"):
        p = baselines.predict(x, kind)
        report[kind] = {str(d): metrics(p[depths == d], y[depths == d]) for d in range(1, 7)}
    return report


def oracle_diagnostics(predictor, records):
    """Evaluator-only full-table means/minima; never supplied to fitting/search.

    Equal-prefix metrics, separately by depth, complement outcome-weighted test
    residuals. This uses ALL outcomes and is explicitly NOT held-out validation.
    """
    sums, counts, minima = defaultdict(float), defaultdict(int), {}
    for row in records:
        state, error = tuple(row["state"]), row["error"]
        for depth in range(7):
            prefix = state[:depth]
            sums[prefix] += error
            counts[prefix] += 1
            minima[prefix] = min(error, minima.get(prefix, float("inf")))
    prefixes = sorted(sums, key=lambda p: (len(p), p))
    predictions = predictor.predict(prefixes)
    pred = dict(zip(prefixes, predictions, strict=True))
    means = {p: sums[p] / counts[p] for p in prefixes}
    by_depth = {}
    for depth in range(1, 7):
        selected = [p for p in prefixes if len(p) == depth]
        by_depth[str(depth)] = metrics([pred[p] for p in selected], [means[p] for p in selected])
    correct_mean, correct_min, pair_correct, pairs, residuals = 0, 0, 0, 0, []
    parents = [p for p in prefixes if len(p) < 6]
    for parent in parents:
        children = [parent + (i,) for i in range(5)]
        chosen = min(children, key=lambda c: pred[c])
        correct_mean += means[chosen] <= min(means[c] for c in children) + 1e-10
        correct_min += minima[chosen] <= min(minima[c] for c in children) + 1e-10
        residuals.append(abs(pred[parent] - sum(pred[c] for c in children) / 5))
        for i in range(5):
            for j in range(i + 1, 5):
                a, b = children[i], children[j]
                difference = means[a] - means[b]
                if abs(difference) > 1e-10:
                    pairs += 1
                    sign = (pred[a] - pred[b]) * difference
                    pair_correct += 1 if sign > 0 else 0.5 if sign == 0 else 0
    return {
        "scope": "POSTHOC ALL-OUTCOME ORACLE, not held-out model selection",
        "mean_prediction_by_depth": by_depth,
        "sibling_best_mean_accuracy": correct_mean / len(parents),
        "sibling_best_minimum_accuracy": correct_min / len(parents),
        "sibling_pair_accuracy": pair_correct / pairs if pairs else None,
        "parent_child_consistency_mae_pp": float(np.mean(residuals)),
    }
