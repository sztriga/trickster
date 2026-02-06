from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np


Activation = Literal["relu", "tanh"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


@dataclass
class MLPBinaryModel:
    """
    N-hidden-layer MLP trained online with SGD on log-loss.

    All hidden layers share the same width (hidden_units).
    Input is a sparse dict of features; the model internally encodes into a dense vector
    using a fixed feature key list.
    """

    feature_keys: List[str]
    hidden_units: int = 64
    hidden_layers: int = 1
    activation: Activation = "relu"
    seed: int = 0

    def __post_init__(self) -> None:
        self._key_to_idx: dict[str, int] = {k: i for i, k in enumerate(self.feature_keys)}
        d = len(self.feature_keys)
        h = int(self.hidden_units)
        L = max(1, int(self.hidden_layers))
        rng = np.random.default_rng(int(self.seed))

        # He init for relu, Xavier for tanh
        def _scale(fan_in: int) -> float:
            if self.activation == "relu":
                return float(np.sqrt(2.0 / max(1, fan_in)))
            return float(np.sqrt(1.0 / max(1, fan_in)))

        # Hidden layers: Ws[i] is (h, fan_in), bs[i] is (h,)
        self.Ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        for i in range(L):
            fan_in = d if i == 0 else h
            s = _scale(fan_in)
            self.Ws.append((rng.standard_normal((h, fan_in)) * s).astype(np.float64))
            self.bs.append(np.zeros((h,), dtype=np.float64))

        # Output layer: (h,) -> scalar
        s = _scale(h)
        self.Wout = (rng.standard_normal((h,)) * s).astype(np.float64)
        self.bout = float(0.0)

    def _encode(self, features: Dict[str, float]) -> np.ndarray:
        x = np.zeros((len(self.feature_keys),), dtype=np.float64)
        for k, v in features.items():
            if not v:
                continue
            idx = self._key_to_idx.get(k)
            if idx is not None:
                x[idx] = float(v)
        return x

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        return np.tanh(z)

    def _act_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(np.float64)
        t = np.tanh(z)
        return (1.0 - t * t).astype(np.float64)

    def predict_proba(self, features: Dict[str, float]) -> float:
        x = self._encode(features)
        a = x
        for W, b in zip(self.Ws, self.bs):
            with np.errstate(all="ignore"):
                z = W @ a + b
            if not np.all(np.isfinite(z)):
                return 0.5
            a = self._act(z)
        with np.errstate(all="ignore"):
            logit = float(self.Wout @ a + self.bout)
        return float(_sigmoid(np.array([logit], dtype=np.float64))[0])

    def predict_proba_batch(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """Batched inference: forward-pass N samples in one matrix multiply.

        Returns an (N,) array of probabilities.  Much faster than calling
        predict_proba in a loop when N > 1.
        """
        N = len(features_list)
        if N == 0:
            return np.empty(0, dtype=np.float64)
        if N == 1:
            return np.array([self.predict_proba(features_list[0])], dtype=np.float64)

        X = self._encode_batch(features_list)           # (N, d)
        A = X
        for W, b in zip(self.Ws, self.bs):
            with np.errstate(all="ignore"):
                Z = A @ W.T + b                          # (N, h)
            if not np.all(np.isfinite(Z)):
                return np.full(N, 0.5, dtype=np.float64)
            A = self._act(Z)                             # (N, h)
        with np.errstate(all="ignore"):
            logits = A @ self.Wout + self.bout           # (N,)
        return _sigmoid(logits)                          # (N,)

    # ------------------------------------------------------------------
    #  Single-sample update (kept for online / direct training)
    # ------------------------------------------------------------------

    def update(self, features: Dict[str, float], y: float, lr: float = 0.05, l2: float = 1e-6) -> float:
        """One SGD step. Returns p(before update)."""
        x = self._encode(features)

        # --- Forward pass: store pre-activations (zs) and activations (hs) ---
        zs: list[np.ndarray] = []
        hs: list[np.ndarray] = []  # hs[i] = act(zs[i]), hs[-1] is input to output layer
        a = x
        for W, b in zip(self.Ws, self.bs):
            with np.errstate(all="ignore"):
                z = W @ a + b
            if not np.all(np.isfinite(z)):
                self.__post_init__()
                return 0.5
            zs.append(z)
            a = self._act(z)
            hs.append(a)

        with np.errstate(all="ignore"):
            logit = float(self.Wout @ hs[-1] + self.bout)
        p = float(_sigmoid(np.array([logit], dtype=np.float64))[0])
        err = float(y - p)

        # --- Backward pass ---
        dWout = err * hs[-1]
        dbout = err
        da = err * self.Wout

        L = len(self.Ws)
        dWs: list[np.ndarray] = [np.empty(0)] * L
        dbs: list[np.ndarray] = [np.empty(0)] * L
        for i in range(L - 1, -1, -1):
            dz = da * self._act_grad(zs[i])
            prev_a = hs[i - 1] if i > 0 else x
            dWs[i] = np.outer(dz, prev_a)
            dbs[i] = dz
            if i > 0:
                da = self.Ws[i].T @ dz

        self._apply_grads(dWout, dbout, dWs, dbs, lr, l2)
        return p

    # ------------------------------------------------------------------
    #  Vectorised mini-batch update (for replay-buffer training)
    # ------------------------------------------------------------------

    def _encode_batch(self, features_list: list[Dict[str, float]]) -> np.ndarray:
        """Encode N feature dicts into an (N, d) matrix."""
        N = len(features_list)
        d = len(self.feature_keys)
        X = np.zeros((N, d), dtype=np.float64)
        for i, features in enumerate(features_list):
            for k, v in features.items():
                if not v:
                    continue
                idx = self._key_to_idx.get(k)
                if idx is not None:
                    X[i, idx] = float(v)
        return X

    def batch_update(
        self,
        features_list: list[Dict[str, float]],
        ys: list[float],
        lr: float = 0.05,
        l2: float = 1e-6,
    ) -> None:
        """Mini-batch SGD step: forward/backward the entire batch as matrices."""
        N = len(features_list)
        if N == 0:
            return

        X = self._encode_batch(features_list)       # (N, d)
        Y = np.array(ys, dtype=np.float64)           # (N,)

        # --- Batched forward pass ---
        Zs: list[np.ndarray] = []     # pre-activations per layer
        As: list[np.ndarray] = [X]    # As[0] = input, As[i+1] = act(Zs[i])
        A = X
        for W, b in zip(self.Ws, self.bs):
            with np.errstate(all="ignore"):
                Z = A @ W.T + b                      # (N, h)
            if not np.all(np.isfinite(Z)):
                self.__post_init__()
                return
            Zs.append(Z)
            A = self._act(Z)                          # (N, h)
            As.append(A)

        with np.errstate(all="ignore"):
            logits = As[-1] @ self.Wout + self.bout   # (N,)
        P = _sigmoid(logits)                          # (N,)

        # --- Error signal ---
        errs = Y - P                                  # (N,)

        # --- Batched backward pass (average gradient over batch) ---
        # Output layer
        dWout = errs @ As[-1] / N                     # (h,)
        dbout = float(np.mean(errs))

        # Gradient w.r.t. last hidden activation
        dA = errs[:, None] * self.Wout[None, :]       # (N, h)

        # Hidden layers (reverse order)
        L = len(self.Ws)
        dWs: list[np.ndarray] = [np.empty(0)] * L
        dbs: list[np.ndarray] = [np.empty(0)] * L
        for i in range(L - 1, -1, -1):
            dZ = dA * self._act_grad(Zs[i])           # (N, h)
            dWs[i] = dZ.T @ As[i] / N                 # (h, fan_in)
            dbs[i] = np.mean(dZ, axis=0)              # (h,)
            if i > 0:
                dA = dZ @ self.Ws[i]                   # (N, fan_in_prev)

        self._apply_grads(dWout, dbout, dWs, dbs, lr, l2)

    # ------------------------------------------------------------------
    #  Shared: gradient clipping, parameter update, stability
    # ------------------------------------------------------------------

    def _apply_grads(
        self,
        dWout: np.ndarray,
        dbout: float,
        dWs: list[np.ndarray],
        dbs: list[np.ndarray],
        lr: float,
        l2: float,
    ) -> None:
        # Gradient clipping
        clip = 5.0
        g2 = float(np.sum(dWout * dWout)) + dbout * dbout
        for dW, db in zip(dWs, dbs):
            g2 += float(np.sum(dW * dW)) + float(np.sum(db * db))
        gnorm = float(np.sqrt(g2))
        if np.isfinite(gnorm) and gnorm > clip:
            scale = clip / (gnorm + 1e-8)
            dWout = dWout * scale
            dbout = dbout * scale
            dWs = [dW * scale for dW in dWs]
            dbs = [db * scale for db in dbs]

        # Parameter update with L2 weight decay
        L = len(self.Ws)
        self.Wout = self.Wout * (1.0 - lr * l2) + lr * dWout
        self.bout = float(self.bout + lr * dbout)
        for i in range(L):
            self.Ws[i] = self.Ws[i] * (1.0 - lr * l2) + lr * dWs[i]
            self.bs[i] = self.bs[i] + lr * dbs[i]

        # Stability: nan/inf cleanup and weight clipping
        np.nan_to_num(self.Wout, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        np.clip(self.Wout, -10.0, 10.0, out=self.Wout)
        if not np.isfinite(self.bout):
            self.bout = 0.0
        self.bout = max(-10.0, min(10.0, float(self.bout)))
        for i in range(L):
            np.nan_to_num(self.Ws[i], copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
            np.nan_to_num(self.bs[i], copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
            np.clip(self.Ws[i], -10.0, 10.0, out=self.Ws[i])
            np.clip(self.bs[i], -10.0, 10.0, out=self.bs[i])


# Backward-compatible alias
OneHiddenMLPBinaryModel = MLPBinaryModel
