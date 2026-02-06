from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
    """Vectorised numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


@dataclass(slots=True)
class LinearBinaryModel:
    """
    Online logistic regression on sparse features.

    We interpret predict_proba(features) as "probability this action wins the trick".
    """

    weights: Dict[str, float] = field(default_factory=dict)

    def predict_logit(self, features: Dict[str, float]) -> float:
        s = self.weights.get("__bias__", 0.0)
        for k, v in features.items():
            if not v:
                continue
            s += self.weights.get(k, 0.0) * v
        return s

    def predict_proba(self, features: Dict[str, float]) -> float:
        return _sigmoid(self.predict_logit(features))

    def predict_proba_batch(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """Batched inference: compute P(win) for N feature dicts at once.

        Returns an (N,) numpy array of probabilities.
        """
        N = len(features_list)
        if N == 0:
            return np.empty(0, dtype=np.float64)

        # Gather the union of all active keys once
        w = self.weights
        bias = w.get("__bias__", 0.0)
        logits = np.full(N, bias, dtype=np.float64)
        for i, features in enumerate(features_list):
            s = 0.0
            for k, v in features.items():
                if not v:
                    continue
                wk = w.get(k)
                if wk:
                    s += wk * v
            logits[i] += s
        return _sigmoid_vec(logits)

    def update(
        self,
        features: Dict[str, float],
        y: float,
        lr: float = 0.05,
        l2: float = 1e-6,
    ) -> float:
        """One SGD step on log-loss. Returns prediction *before* the update."""
        p = self.predict_proba(features)
        err = y - p

        self.weights["__bias__"] = self.weights.get("__bias__", 0.0) * (1.0 - lr * l2) + lr * err * 1.0

        for k, x in features.items():
            if not x:
                continue
            w = self.weights.get(k, 0.0)
            w = w * (1.0 - lr * l2) + lr * err * x
            self.weights[k] = w

        return p

    def batch_update(
        self,
        features_list: List[Dict[str, float]],
        ys: List[float],
        lr: float = 0.05,
        l2: float = 1e-6,
    ) -> None:
        """Mini-batch SGD step: accumulate average gradient, apply once."""
        N = len(features_list)
        if N == 0:
            return

        # Accumulate average gradient over the batch
        grad: dict[str, float] = {}
        for features, y in zip(features_list, ys):
            p = self.predict_proba(features)
            err = y - p
            grad["__bias__"] = grad.get("__bias__", 0.0) + err
            for k, x in features.items():
                if not x:
                    continue
                grad[k] = grad.get(k, 0.0) + err * x

        inv_n = 1.0 / N
        decay = 1.0 - lr * l2

        # Apply averaged gradient
        self.weights["__bias__"] = self.weights.get("__bias__", 0.0) * decay + lr * grad.get("__bias__", 0.0) * inv_n
        for k, g in grad.items():
            if k == "__bias__":
                continue
            w = self.weights.get(k, 0.0)
            self.weights[k] = w * decay + lr * g * inv_n

