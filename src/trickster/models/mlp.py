from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np


Activation = Literal["relu", "tanh"]

# ---------------------------------------------------------------------------
# Numerically stable helpers
# ---------------------------------------------------------------------------

# Clamp hidden activations to prevent cascading overflow through layers.
_ACT_CLIP = 20.0
# Clamp logits before sigmoid to avoid exp() overflow / log-loss edge cases.
_LOGIT_CLIP = 15.0
# Maximum allowed magnitude for any single weight or bias.
_WEIGHT_CLIP = 5.0
# Global gradient-norm clip threshold.
_GRAD_CLIP = 5.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable vectorised sigmoid."""
    x = np.clip(x, -_LOGIT_CLIP, _LOGIT_CLIP)
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

        # Xavier uniform for all layers — works well for both relu and tanh
        # at this model size, and produces smaller initial weights than He.
        def _limit(fan_in: int, fan_out: int) -> float:
            return float(np.sqrt(6.0 / max(1, fan_in + fan_out)))

        # Hidden layers: Ws[i] is (h, fan_in), bs[i] is (h,)
        self.Ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        for i in range(L):
            fan_in = d if i == 0 else h
            lim = _limit(fan_in, h)
            self.Ws.append(rng.uniform(-lim, lim, (h, fan_in)).astype(np.float64))
            self.bs.append(np.zeros((h,), dtype=np.float64))

        # Output layer: (h,) -> scalar — small init keeps initial logits near 0
        lim = _limit(h, 1)
        self.Wout = rng.uniform(-lim, lim, (h,)).astype(np.float64)
        self.bout = float(0.0)

    # ------------------------------------------------------------------
    #  Forward helpers (with activation clipping)
    # ------------------------------------------------------------------

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
            return np.minimum(np.maximum(z, 0.0), _ACT_CLIP)
        return np.tanh(z)  # tanh is already bounded in [-1, 1]

    def _act_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return ((z > 0) & (z < _ACT_CLIP)).astype(np.float64)
        t = np.tanh(z)
        return (1.0 - t * t).astype(np.float64)

    def _forward(self, A: np.ndarray):
        """Batched forward pass.  A is (N, d) or (d,).

        Returns (probabilities, list_of_pre_activations, list_of_activations)
        where activations[0] = input A, activations[i+1] = act(Z_i).

        Suppresses spurious numpy/BLAS warnings from sparse inputs — the
        activation clipping + logit clipping guarantee bounded outputs.
        """
        Zs: list[np.ndarray] = []
        As: list[np.ndarray] = [A]
        with np.errstate(all="ignore"):
            for W, b in zip(self.Ws, self.bs):
                Z = A @ W.T + b
                Zs.append(Z)
                A = self._act(Z)
                As.append(A)
            logits = np.clip(As[-1] @ self.Wout + self.bout, -_LOGIT_CLIP, _LOGIT_CLIP)
        P = _sigmoid(logits)
        return P, Zs, As

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: Dict[str, float]) -> float:
        x = self._encode(features)
        P, _, _ = self._forward(x)
        return float(P) if P.ndim == 0 else float(P[0])

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
        P, _, _ = self._forward(X)
        return P

    def forward_raw(self, X: np.ndarray) -> np.ndarray:
        """Forward pass on a pre-encoded (N, d) numpy array.

        Skips all dict -> array conversion.  Returns (N,) probabilities.
        Use with :class:`FastEncoder` for maximum throughput in hot loops.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        P, _, _ = self._forward(X)
        return P

    # ------------------------------------------------------------------
    #  Single-sample update (kept for online / direct training)
    # ------------------------------------------------------------------

    def update(self, features: Dict[str, float], y: float, lr: float = 0.05, l2: float = 1e-6) -> float:
        """One SGD step. Returns p(before update)."""
        x = self._encode(features)
        P, Zs, As = self._forward(x)
        p = float(P) if P.ndim == 0 else float(P[0])
        err = float(y - p)

        # --- Backward pass ---
        with np.errstate(all="ignore"):
            dWout = err * As[-1]
            dbout = err
            da = err * self.Wout

            L = len(self.Ws)
            dWs: list[np.ndarray] = [np.empty(0)] * L
            dbs: list[np.ndarray] = [np.empty(0)] * L
            for i in range(L - 1, -1, -1):
                dz = da * self._act_grad(Zs[i])
                prev_a = As[i]  # As[0] = input x
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

        # --- Batched forward pass (activation-clipped, no overflow) ---
        P, Zs, As = self._forward(X)

        # --- Error signal ---
        errs = Y - P                                  # (N,)

        # --- Batched backward pass (average gradient over batch) ---
        # errstate: sparse relu activations cause harmless BLAS warnings.
        with np.errstate(all="ignore"):
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

    def batch_update_raw(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lr: float = 0.05,
        l2: float = 1e-6,
    ) -> float:
        """Batch SGD on pre-encoded numpy arrays.  X: (N, d), Y: (N,).

        Identical to :meth:`batch_update` but skips dict→array encoding.
        Returns the mean log-loss *before* the update.
        """
        N = X.shape[0]
        if N == 0:
            return 0.0

        P, Zs, As = self._forward(X)
        errs = Y - P  # (N,)

        # Mean log-loss (for monitoring)
        eps = 1e-12
        loss = float(-np.mean(Y * np.log(np.clip(P, eps, 1.0))
                              + (1.0 - Y) * np.log(np.clip(1.0 - P, eps, 1.0))))

        with np.errstate(all="ignore"):
            dWout = errs @ As[-1] / N
            dbout = float(np.mean(errs))
            dA = errs[:, None] * self.Wout[None, :]
            L = len(self.Ws)
            dWs: list[np.ndarray] = [np.empty(0)] * L
            dbs: list[np.ndarray] = [np.empty(0)] * L
            for i in range(L - 1, -1, -1):
                dZ = dA * self._act_grad(Zs[i])
                dWs[i] = dZ.T @ As[i] / N
                dbs[i] = np.mean(dZ, axis=0)
                if i > 0:
                    dA = dZ @ self.Ws[i]

        self._apply_grads(dWout, dbout, dWs, dbs, lr, l2)
        return loss

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
        # Compute global gradient norm
        g2 = float(np.sum(dWout * dWout)) + dbout * dbout
        for dW, db in zip(dWs, dbs):
            g2 += float(np.sum(dW * dW)) + float(np.sum(db * db))
        gnorm = float(np.sqrt(g2))

        # If gradient is NaN/Inf, skip this update entirely (don't corrupt weights)
        if not np.isfinite(gnorm):
            return

        # Gradient clipping
        if gnorm > _GRAD_CLIP:
            scale = _GRAD_CLIP / (gnorm + 1e-8)
            dWout = dWout * scale
            dbout = dbout * scale
            dWs = [dW * scale for dW in dWs]
            dbs = [db * scale for db in dbs]

        # Parameter update with L2 weight decay
        decay = 1.0 - lr * l2
        L = len(self.Ws)
        self.Wout = self.Wout * decay + lr * dWout
        self.bout = float(self.bout + lr * dbout)
        for i in range(L):
            self.Ws[i] = self.Ws[i] * decay + lr * dWs[i]
            self.bs[i] = self.bs[i] + lr * dbs[i]

        # Hard weight clip — last line of defence
        np.clip(self.Wout, -_WEIGHT_CLIP, _WEIGHT_CLIP, out=self.Wout)
        self.bout = max(-_WEIGHT_CLIP, min(_WEIGHT_CLIP, float(self.bout)))
        for i in range(L):
            np.clip(self.Ws[i], -_WEIGHT_CLIP, _WEIGHT_CLIP, out=self.Ws[i])
            np.clip(self.bs[i], -_WEIGHT_CLIP, _WEIGHT_CLIP, out=self.bs[i])


# Backward-compatible alias
OneHiddenMLPBinaryModel = MLPBinaryModel
